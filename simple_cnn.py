# simple_cnn_classification.py
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Config / Hyperparams
# -------------------------
DATA_DIR = "dataset_classification"   # ubah sesuai folder
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

IMG_SIZE = 128       # ubah ke 224/256 jika GPU; 128 aman untuk CPU
BATCH_SIZE = 16      # kecilkan jadi 8 atau 4 kalau CPU lambat
NUM_EPOCHS = 20
LR = 1e-3
NUM_WORKERS = 0      # di Windows set 0; di Linux bisa >0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "simple_cnn_best.pth"


# -------------------------
# 1) Transforms & DataLoaders
# -------------------------
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset = datasets.ImageFolder(VAL_DIR,   transform=val_transforms)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS)

class_names = train_dataset.classes
num_classes = len(class_names)
print("Classes:", class_names)
print("Device:", DEVICE)


# -------------------------
# 2) Simple CNN Model (from scratch)
# -------------------------
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, dropout=0.25):
        super().__init__()
        # Convolutional part (feature extractor)
        # Conv block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3,
                      padding=1),  # out: 32 x H x W
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # downsample x2
        )
        # Conv block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # out: 64 x H/2 x W/2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # Conv block 3
        self.conv3 = nn.Sequential(
            # out: 128 x H/4 x W/4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Global adaptive pooling -> flattens spatial dims regardless of input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (4, 4))  # reduce to known size

        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)   # conv -> relu -> pool
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adaptive_pool(x)   # ensures fixed-size vector before FC
        x = self.classifier(x)
        return x


# -------------------------
# 3) Train / Validate Helpers
# -------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == targets).sum().item()
        total += imgs.size(0)

    epoch_loss = running_loss / total
    acc = correct / total
    return epoch_loss, acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
            total += imgs.size(0)

    epoch_loss = running_loss / total
    acc = correct / total
    return epoch_loss, acc


# -------------------------
# 4) Main training loop
# -------------------------
def run_training():
    model = SimpleCNN(in_channels=3, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}  "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}  "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
            }, MODEL_PATH)
            print(f"Saved best model (val_acc={val_acc:.4f}) -> {MODEL_PATH}")

    elapsed = time.time() - start_time
    print("Training finished in: {:.1f} seconds".format(elapsed))
    print("Best val acc:", best_val_acc)


# -------------------------
# 5) Inference + visualize
# -------------------------
def predict_image(model_path, image_path, topk=3):
    # load model with same architecture
    model = SimpleCNN(in_channels=3, num_classes=num_classes).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # preprocess image
    tf = val_transforms
    img = torchvision.io.read_image(
        image_path).float() / 255.0  # C,H,W with 0-1
    img = transforms.functional.resize(img, (IMG_SIZE, IMG_SIZE))
    img = transforms.functional.normalize(
        img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = img.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(img)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]

    topk_idx = np.argsort(probs)[-topk:][::-1]
    for i in topk_idx:
        print(f"{class_names[i]}: {probs[i]:.4f}")

    # show image
    img_np = torchvision.io.read_image(
        image_path).permute(1, 2, 0).numpy() / 255.0
    plt.imshow(img_np)
    plt.title(f"Pred: {class_names[topk_idx[0]]} ({probs[topk_idx[0]]:.2f})")
    plt.axis("off")
    plt.show()


# -------------------------
# 6) Run if script
# -------------------------
if __name__ == "__main__":
    run_training()
    # after training run example:
    # predict_image(MODEL_PATH, "dataset_classification/val/classA/some_image.jpg")
