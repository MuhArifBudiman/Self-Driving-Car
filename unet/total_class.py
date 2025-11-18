import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def detect_classes(mask_dir) -> int:
    unique_values = set()

    mask_files = [f for f in os.listdir(
        mask_dir) if f.endswith((".png", ".jpg"))]

    print(f"🔍 Found {len(mask_files)} mask files.")

    for file in tqdm(mask_files, desc="Processing masks"):
        mask_path = os.path.join(mask_dir, file)

        mask = np.array(Image.open(mask_path))

        # Tambahkan semua nilai pixel baru ke set
        unique_values.update(np.unique(mask))

    print(f"Total Classes: {len(sorted(list(unique_values)))}")
    return sorted(list(unique_values))


if __name__ == "__main__":
    MASK_PATH = "dataset_cityscapes/train/masks"

    classes = detect_classes(MASK_PATH)

    print("\n📌 Unique Classes Found:", classes)
    print(f"🎯 Total Classes: {len(classes)}")
