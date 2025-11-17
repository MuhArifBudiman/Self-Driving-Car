import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor
from tqdm import tqdm
import cv2
import numpy as np
import os


# ------------------ Dataset Loader -----------------------
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        # Load image
        img = cv2.imread(os.path.join(self.img_dir, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = ToTensor()(img)

        # Load mask
        mask_path = os.path.join(
            self.mask_dir, img_name.replace(".jpg", ".png"))
        mask = cv2.imread(mask_path, 0)

        # Detect instance IDs
        ids = np.unique(mask)
        ids = ids[ids != 0]

        # Create mask tensor: K x H x W
        masks = (mask[None] == ids[:, None, None]).astype(np.uint8)

        # Bounding boxes (xmin, ymin, xmax, ymax)
        boxes = []
        for m in masks:
            pos = np.where(m)
            xmin, xmax = pos[1].min(), pos[1].max()
            ymin, ymax = pos[0].min(), pos[0].max()
            boxes.append([xmin, ymin, xmax, ymax])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.ones((len(ids),), dtype=torch.int64),  # class = 1
            "masks": torch.tensor(masks, dtype=torch.uint8)
        }

        return img, target


# ------------------ Load Mask R-CNN Model -----------------------
def get_model(num_classes=2):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights="COCO_V1")

    # Replace classifier head
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_channels=1024, num_classes=num_classes
    )

    # Replace mask head
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_channels=256, dim_reduced=256, num_classes=num_classes
    )

    return model


# ------------------ Training Function -------------------------
def train_model(dataset_path, epochs=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Training on: {device.upper()}")

    train_data = SegmentationDataset(
        img_dir=os.path.join(dataset_path, "train/images"),
        mask_dir=os.path.join(dataset_path, "train/masks")
    )
    val_data = SegmentationDataset(
        img_dir=os.path.join(dataset_path, "val/images"),
        mask_dir=os.path.join(dataset_path, "val/masks")
    )

    train_loader = DataLoader(train_data, batch_size=2,
                              shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_data, batch_size=2,
                            shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    model = get_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    best_loss = float("inf")

    print("\n📌 Training started...\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        train_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} (Training)", ncols=90)

        for imgs, targets in train_bar:
            imgs = [i.to(device) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]

            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.train()
        val_loss = 0

        val_bar = tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{epochs} (Validating)", ncols=90)

        with torch.no_grad():
            for imgs, targets in val_bar:
                imgs = [i.to(device) for i in imgs]
                targets = [{k: v.to(device) for k, v in t.items()}
                           for t in targets]

                loss_dict = model(imgs, targets)
                val_loss += sum(loss_dict.values()).item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

        # ---- SAVE BEST MODEL ----
        if avg_loss < best_loss:
            best_loss = avg_loss
            result_path = 'result_model'
            os.makedirs(result_path, exist_ok=True)
            if epochs < 5:
                print("Testing model berhasil, tentukan epochs")
            else:
                torch.save(model.state_dict(), os.path.join(
                    result_path, "best_maskrcnn_model.pth"))
                print("💾 Model improved → saved!")

    print("\n🎉 Training Finished! Best model saved as: best_maskrcnn_model.pth")


# ------------------ MAIN -------------------------
if __name__ == "__main__":
    dataset_path = "dataset"
    train_model(dataset_path, epochs=20)
