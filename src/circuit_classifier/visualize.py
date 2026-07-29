"""Plotting helpers for dataset samples and training history."""

import os

import cv2
import matplotlib.pyplot as plt


def visualize_samples(data_path, class_names, samples_per_class=3):
    """Show a grid of sample images for the first few classes."""
    n_rows = min(len(class_names), 5)
    fig, axes = plt.subplots(
        n_rows,
        samples_per_class,
        figsize=(samples_per_class * 3, n_rows * 3),
    )

    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif samples_per_class == 1:
        axes = axes.reshape(-1, 1)

    for idx, class_name in enumerate(class_names[:5]):
        class_path = os.path.join(data_path, class_name)
        images = [
            f
            for f in os.listdir(class_path)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ][:samples_per_class]

        for j, img_name in enumerate(images):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            axes[idx, j].imshow(img, cmap="gray")
            axes[idx, j].axis("off")
            if j == 0:
                axes[idx, j].set_title(class_name, fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    """Plot training/validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Training Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["loss"], label="Training Loss")
    axes[1].plot(history.history["val_loss"], label="Validation Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
