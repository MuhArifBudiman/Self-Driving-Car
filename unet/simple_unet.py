import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import cv2
import numpy as np
import os

# ---- Albumentations untuk augmentasi ----
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ------------------ Dataset Loader -----------------------
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

        print("📌 Kelas terdeteksi:", self.value_to_index)
        # -------------------------

        # Transform + augmentation
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1,
                               rotate_limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            ToTensorV2()
        ])

        # Transform only resize → no augmentation (untuk validation)
        self.transform_no_aug = A.Compose([
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

        # -------------------------
        # 🔥 Convert grayscale → class label index
        # -------------------------
        mask = np.vectorize(self.value_to_index.get)(mask)
        # print(np.unique(mask))
        return img, torch.tensor(mask, dtype=torch.long)


# ------------------ UNET MODEL -----------------------
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


# ------------------ TRAINING LOOP -----------------------
def train_model(dataset_path, epochs=20):
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

    model = UNet(num_classes=12).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")

    print("\n📌 Training started...\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        train_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} (Training)", ncols=90)
        try:
            for imgs, masks in train_bar:
                imgs, masks = imgs.to(device), masks.to(device)

                preds = model(imgs)
                loss = criterion(preds, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
        except ValueError as ve:
            print(ve)

        avg_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            val_bar = tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{epochs} Validating", ncols=90)
            for imgs, masks in val_bar:
                imgs, masks = imgs.to(device), masks.to(device)

                preds = model(imgs)
                loss = criterion(preds, masks)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs("result_unet", exist_ok=True)
            torch.save(model.state_dict(), "result_unet/best_unet_model.pth")
            print("💾 Model improved → saved!")

    print("\n🎉 Training Finished! Best model saved as: result_unet/best_unet_model.pth")


# ------------------ MAIN -----------------------
if __name__ == "__main__":
    dataset_path = "dataset"
    train_model(dataset_path, epochs=20)
