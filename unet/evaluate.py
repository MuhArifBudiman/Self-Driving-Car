import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_training_history(history):
    """
        Membuat dua plot: 
        1. Loss Pelatihan
        2. Akurasi Pelatihan vs. Validasi
    """

    epochs = range(1, len(history['train_loss']) + 1)

    # --- Plot 1: Loss Pelatihan ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)  # 1 baris, 2 kolom, plot ke-1
    plt.plot(epochs, history['train_loss'], 'b', label='Training Loss')

    plt.plot(epochs, history['val_loss'], 'g', label='Validation Accuracy')
    plt.title('Training vs Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # --- Plot 2: Akurasi Pelatihan vs. Validasi ---
    plt.subplot(1, 2, 2)  # 1 baris, 2 kolom, plot ke-2
    plt.plot(epochs, history['train_acc'], 'r', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'g', label='Validation Accuracy')
    plt.title('Training vs. Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()  # Mengatur jarak antar subplot
    plt.savefig('acc_loss_epochs.png')
    print("Saved history")


def calculate_miou(preds, masks, num_classes):
    """Compute mIoU per batch"""

    preds = torch.argmax(preds, dim=1).cpu().numpy()
    masks = masks.cpu().numpy()

    ious = []
    for cls in range(num_classes):
        pred_cls = preds == cls
        true_cls = masks == cls

        intersection = (pred_cls & true_cls).sum()
        union = (pred_cls | true_cls).sum()

        if union == 0:
            continue  # skip class not in this batch

        ious.append(intersection / union)

    return np.mean(ious) if len(ious) > 0 else 0
