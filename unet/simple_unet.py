import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from torchvision import models

from sklearn.utils.class_weight import compute_class_weight

from tqdm import tqdm
import cv2
import numpy as np
from time import time
from datetime import datetime

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .total_class import detect_classes
from .evaluate import calculate_miou, plot_training_history

import os

CURRENT_DIR = os.getcwd()


def gpu_usage():
    if torch.cuda.is_available():
        alloc = round(torch.cuda.memory_allocated() / 1024**2, 1)
        reserved = round(torch.cuda.memory_reserved() / 1024**2, 1)
        return f"GPU: {alloc}MB allocated / {reserved}MB reserved"
    return "CPU Mode"


class SegmentationDataset(Dataset):

    def __init__(self, img_dir, mask_dir, augment=False, resize_dim=(512, 1024)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))
        self.resize_dim = resize_dim
        self.augment = augment

        all_mask_values = set()
        for file in os.listdir(mask_dir):
            mask = cv2.imread(os.path.join(mask_dir, file), 0)
            all_mask_values.update(np.unique(mask))

        self.mask_values = sorted(list(all_mask_values))
        self.value_to_index = {v: i for i, v in enumerate(self.mask_values)}

        self.normalize = A.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))

        # Transform + augmentation
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1,
                               rotate_limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            self.normalize,
            ToTensorV2()
        ])

        self.transform_no_aug = A.Compose([
            self.normalize,
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(
            self.mask_dir, img_name.replace(".jpg", ".png"))

        # Load image (BGR→RGB)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load mask
        mask = cv2.imread(mask_path, 0)

        # ---- Resize agar divisible by 16 ----
        img = cv2.resize(img, self.resize_dim)
        mask = cv2.resize(mask, self.resize_dim,
                          interpolation=cv2.INTER_NEAREST)

        # ---- Apply augmentation (training only) ----
        if self.augment:
            transformed = self.transform(image=img, mask=mask)
        else:
            transformed = self.transform_no_aug(image=img, mask=mask)

        img = transformed["image"].float()
        mask = transformed["mask"]

        mask = np.vectorize(self.value_to_index.get)(mask)

        return img, torch.tensor(mask, dtype=torch.long)


class UNet(nn.Module):

    def __init__(self, num_classes=12):
        super(UNet, self).__init__()

        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = block(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.encoder4 = block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = block(128, 64)

        self.final_layer = nn.Conv2d(64, num_classes, kernel_size=1)

    def crop_to_match(self, enc, dec):
        _, _, h, w = dec.shape
        return enc[:, :, :h, :w]

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.upconv4(b)
        e4 = self.crop_to_match(e4, d4)
        d4 = self.decoder4(torch.cat([d4, e4], dim=1))

        d3 = self.upconv3(d4)
        e3 = self.crop_to_match(e3, d3)
        d3 = self.decoder3(torch.cat([d3, e3], dim=1))

        d2 = self.upconv2(d3)
        e2 = self.crop_to_match(e2, d2)
        d2 = self.decoder2(torch.cat([d2, e2], dim=1))

        d1 = self.upconv1(d2)
        e1 = self.crop_to_match(e1, d1)
        d1 = self.decoder1(torch.cat([d1, e1], dim=1))

        return self.final_layer(d1)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, smooth=1e-6):
        preds = torch.softmax(preds, dim=1)
        preds = preds.argmax(dim=1)

        intersection = (preds == targets).sum().float()
        union = preds.numel() + targets.numel()

        return 1 - (2 * intersection + smooth) / (union + smooth)
    
# ------------------ TRAINING LOOP -----------------------

def train_model(dataset_path, num_classes, epochs=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Training on: {device.upper()}")

    train_data = SegmentationDataset(
        img_dir=os.path.join(dataset_path, "train/images"),
        mask_dir=os.path.join(dataset_path, "train/masks"),
        augment=True
    )

    val_data = SegmentationDataset(
        img_dir=os.path.join(dataset_path, "val/images"),
        mask_dir=os.path.join(dataset_path, "val/masks"),
        augment=False
    )

    train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=2, shuffle=False)

    model = UNet(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    mask_dir = os.path.join(dataset_path, "train/masks")

    class_list = detect_classes(mask_dir)   # already sorted from your function

    # 2️⃣ Ensure jumlah kelas sesuai
    assert num_classes == len(class_list), \
        f"Mismatch: num_classes ({num_classes}) ≠ detected mask classes ({len(class_list)})"

    print(f"📌 Using {num_classes} classes")

    # 3️⃣ Mapping pixel values → class index (karena mask mungkin tidak mulai dari 0)
    label_mapping = {label: idx for idx, label in enumerate(class_list)}
    print(f"🗂 Label Mapping: {label_mapping}")

    # 4️⃣ Kumpulkan semua pixel mask untuk compute class weight
    all_pixels = []
    for f in os.listdir(mask_dir):
        m = cv2.imread(os.path.join(mask_dir, f), 0)
        m = np.vectorize(label_mapping.get)(m)
        all_pixels.extend(m.reshape(-1))

    all_pixels = np.array(all_pixels)

    # 5️⃣ Hitung kelas yang benar-benar muncul
    present_classes = np.unique(all_pixels)
    print(
        f"📌 Classes present in dataset: {present_classes.tolist()} ({len(present_classes)} classes used)")

    # 6️⃣ Compute class weight hanya dari kelas yang muncul
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=all_pixels
    )

    # 7️⃣ Buat tensor dengan size FULL num_classes → biar indexing tetap konsisten
    weights_tensor = torch.zeros(num_classes, dtype=torch.float)

    for idx, cw in zip(present_classes, class_weights):
        weights_tensor[idx] = cw

    weights_tensor = weights_tensor.to(device)

    ce_loss = nn.CrossEntropyLoss(weight=weights_tensor)
    dice_loss = DiceLoss()

    # best_loss = float("inf")
    best_miou = 0.0
    scaler = GradScaler()  # torch.cuda.amp.
    train_history = {
        "train_acc": [],
        "train_loss": [],
        "val_acc": [],
        "val_loss": []
    }
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start = time()
    print(f"\n[INFO] Training started on {start_time} \n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_correct_pixels = 0
        train_total_pixels = 0
        val_miou_total = 0
        val_correct_pixels = 0
        val_total_pixels = 0
        train_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} (Training)", ncols=90)

        for batch_idx, (imgs, masks) in enumerate(train_bar):
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()

        # Mixed Precision forward pass
            with autocast('cuda'):
                preds = model(imgs)
                loss = ce_loss(preds, masks) + 0.5 * \
                    dice_loss(preds, masks)
                train_correct_pixels += (torch.argmax(preds,
                                         dim=1) == masks).sum().item()

                train_total_pixels += masks.numel()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        if (batch_idx % 5 == 0) | (batch_idx == 0):
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
            tqdm.write(gpu_usage())

        avg_loss = total_loss / len(train_loader)
        # VALIDATION
        model.eval()
        val_loss = 0

        with torch.no_grad():
            val_bar = tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{epochs} Validating", ncols=90)
            for imgs, masks in val_bar:
                imgs, masks = imgs.to(device), masks.to(device)

                preds = model(imgs)
                loss = ce_loss(preds, masks) + 0.5 * dice_loss(preds, masks)

                miou = calculate_miou(preds, masks, num_classes)
                val_miou_total += miou
                val_correct_pixels += (torch.argmax(preds,
                                       dim=1) == masks).sum().item()
                val_total_pixels += masks.numel()
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_miou = val_miou_total / len(val_loader)
        train_accuracy = train_correct_pixels / train_total_pixels
        val_accuracy = val_correct_pixels / val_total_pixels

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        train_history['train_acc'].append(train_accuracy)
        train_history['train_loss'].append(avg_loss)
        train_history['val_acc'].append(val_accuracy)
        train_history['val_loss'].append(avg_val_loss)

        print(f"[INFO] LR now: {current_lr:.8f}")

        if avg_val_miou < best_miou:
            best_miou = avg_val_miou
            os.makedirs("result_unet", exist_ok=True)
            torch.save(model.state_dict(), "result_unet/best_unet_model.pth")
            print(f"💾 Model improved → saved! with Best mIoU: {best_miou}")

    plot_training_history(history=train_history)

    finish_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    finish = time()
    elapsed = finish - start

    print(f"\n[INFO] Training Finished on {finish_time} | Training time {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)\n \
          Best model saved as: result_unet/best_unet_model.pth")


# ------------------ MAIN -----------------------
if __name__ == "__main__":

    dataset_path = "dataset_cityscapes"
    masks_path = os.path.join(CURRENT_DIR, dataset_path, "train", "masks")
    total_class = len(detect_classes(mask_dir=masks_path))
    train_model(dataset_path, num_classes=total_class, epochs=2)
