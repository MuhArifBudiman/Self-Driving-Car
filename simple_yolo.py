from ultralytics import YOLO

def train_yolov8_seg():
    # Load model YOLOv8n-seg (paling ringan)
    model = YOLO("yolo11s-seg.pt")

    model.train(
        data="dataset_segmentation/data.yaml",
        imgsz=256,       # kecil agar ringan
        epochs=5,        # bisa ditambah nanti
        batch=2,    # CPU only
        device="gpu"     # paksa CPU
    )

    print("\nTraining selesai!")
    print("Model tersimpan di folder runs/segment/train/")

if __name__ == "__main__":
    train_yolov8_seg()