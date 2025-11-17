import os
import numpy as np
import cv2

# ganti sesuai path kamu
mask_dir = "dataset/train/masks"
values = set()

for fname in os.listdir(mask_dir):
    if fname.endswith((".png", ".jpg", ".jpeg")):
        mask = cv2.imread(os.path.join(mask_dir, fname), 0)
        unique = np.unique(mask)
        values.update(unique)
        print(fname, unique)  # tampilkan per file
    if len(values) > 50:  # kalau kebanyakan, stop
        break

print("\nSEMUA NILAI MASK DITEMUKAN:", sorted(values))
