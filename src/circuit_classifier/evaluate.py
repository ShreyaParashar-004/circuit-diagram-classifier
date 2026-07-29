"""
Evaluate a trained model on the validation set: accuracy, confusion
matrix, and classification report.

Usage:
    python -m circuit_classifier.evaluate --model-path ./models/best_circuit_model.h5
"""

import argparse
import os
import sys

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
import config as cfg  # noqa: E402

from circuit_classifier.data import prepare_data_structure  # noqa: E402
from circuit_classifier.generators import create_eval_generator  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate circuit classifier")
    parser.add_argument("--model-path", default=cfg.BEST_MODEL_PATH)
    parser.add_argument("--dataset-path", default=cfg.RAW_DATASET_PATH)
    parser.add_argument("--organized-path", default=cfg.ORGANIZED_DATA_PATH)
    return parser.parse_args()


def main():
    args = parse_args()

    _, class_names = prepare_data_structure(args.dataset_path, args.organized_path)
    # shuffle=False here so predictions line up with val_gen.classes below —
    # see create_eval_generator's docstring for why this matters.
    val_gen = create_eval_generator(args.organized_path, cfg.CONFIG)

    model = tf.keras.models.load_model(args.model_path)

    val_loss, val_acc, val_top_k = model.evaluate(val_gen, verbose=0)
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Top-3 Accuracy: {val_top_k:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")

    val_gen.reset()
    predictions = model.predict(val_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_gen.classes

    class_labels = list(val_gen.class_indices.keys())
    cm = confusion_matrix(true_classes, predicted_classes)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))


if __name__ == "__main__":
    main()
