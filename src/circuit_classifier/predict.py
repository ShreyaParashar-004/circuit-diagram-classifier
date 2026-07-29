"""
Run inference on a single image or a folder of images.

Usage:
    python -m circuit_classifier.predict --image path/to/image.png
    python -m circuit_classifier.predict --image-dir path/to/folder
"""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
import config as cfg  # noqa: E402

from circuit_classifier.data import get_classes_info  # noqa: E402
from circuit_classifier.preprocessing import preprocess_image_version2  # noqa: E402


def predict_single_image(model, img_path, class_names, img_size, preprocess=True):
    """Predict the class of a single image, returning class + confidence
    + top-3 predictions."""
    if preprocess:
        img = preprocess_image_version2(img_path, target_size=img_size)
    else:
        import cv2

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size)
        img = img.astype(np.float32) / 255.0

    img_batch = np.expand_dims(img, axis=0)
    img_batch = np.expand_dims(img_batch, axis=-1)

    predictions = model.predict(img_batch, verbose=0)[0]
    top_idx = np.argmax(predictions)
    top_3_idx = np.argsort(predictions)[-3:][::-1]

    return {
        "predicted_class": class_names[top_idx],
        "confidence": float(predictions[top_idx]),
        "top_3": [(class_names[i], float(predictions[i])) for i in top_3_idx],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Predict circuit component class")
    parser.add_argument("--model-path", default=cfg.FINAL_MODEL_PATH)
    parser.add_argument("--dataset-path", default=cfg.RAW_DATASET_PATH,
                         help="Used only to recover class names")
    parser.add_argument("--image", help="Path to a single image")
    parser.add_argument("--image-dir", help="Path to a folder of images")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.image and not args.image_dir:
        raise SystemExit("Provide either --image or --image-dir")

    class_names, _ = get_classes_info(args.dataset_path)
    model = tf.keras.models.load_model(args.model_path)
    img_size = cfg.CONFIG["img_size"]

    image_paths = (
        [args.image]
        if args.image
        else [
            os.path.join(args.image_dir, f) for f in sorted(os.listdir(args.image_dir))
        ]
    )

    for image_path in image_paths:
        result = predict_single_image(model, image_path, class_names, img_size)
        print(f"Image: {image_path}")
        print(f"Predicted: {result['predicted_class']} ({result['confidence']:.2%})")
        print(f"Top 3: {result['top_3']}\n")


if __name__ == "__main__":
    main()
