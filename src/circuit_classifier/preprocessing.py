"""
Image preprocessing for hand-drawn circuit component images.

Two preprocessing pipelines are kept from the original experiments:

- ``preprocess_circuit_image``: simpler pipeline (invert + median blur +
  histogram equalization). Used during the initial data exploration.
- ``preprocess_image_version2``: adaptive-threshold based pipeline with
  morphological cleaning. This is the one used for single-image
  inference in ``predict.py``.
"""

import cv2
import numpy as np


def preprocess_circuit_image(img_path, target_size=(64, 64), invert=False):
    """Binarize, denoise, and normalize a circuit image.

    - Converts to white-on-black if the image looks black-on-white.
    - Applies a slight median blur to denoise.
    - Equalizes contrast.
    - Resizes and normalizes to [0, 1].

    Returns a float32 array, or None if the image couldn't be read.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # mean > 127 implies it's probably black-on-white, hence invert
    if invert or img.mean() > 127:
        img = cv2.bitwise_not(img)

    img = cv2.medianBlur(img, 3)
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0

    return img


def preprocess_image_version2(img_path, target_size=(64, 64), show=False):
    """Adaptive-threshold preprocessing pipeline used for inference.

    - Converts to grayscale.
    - Gaussian blur to reduce noise.
    - Adaptive thresholding (white lines on black background).
    - Morphological opening to remove specks.
    - Median blur for edge smoothing.
    - Resizes and normalizes to [0, 1].
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image from {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # white lines on black background
        blockSize=31,
        C=7,
    )

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.medianBlur(binary, 3)

    resized = cv2.resize(binary, target_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0

    if show:
        import matplotlib.pyplot as plt

        plt.imshow(normalized, cmap="gray")
        plt.title("Preprocessed Image")
        plt.axis("off")
        plt.show()

    return normalized
