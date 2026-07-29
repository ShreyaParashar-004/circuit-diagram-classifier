"""
Train the circuit component classifier.

Usage:
    python -m circuit_classifier.train
    python -m circuit_classifier.train --dataset-path ./data/raw/dataset_final --epochs 30
"""

import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "configs"))
import config as cfg  # noqa: E402

from circuit_classifier.data import (  # noqa: E402
    explore_dataset,
    prepare_data_structure,
    print_dataset_summary,
)
from circuit_classifier.generators import create_data_generators  # noqa: E402
from circuit_classifier.model import build_custom_cnn  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train circuit component classifier")
    parser.add_argument("--dataset-path", default=cfg.RAW_DATASET_PATH)
    parser.add_argument("--organized-path", default=cfg.ORGANIZED_DATA_PATH)
    parser.add_argument("--model-dir", default=cfg.MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=cfg.CONFIG["num_epochs"])
    parser.add_argument("--batch-size", type=int, default=cfg.CONFIG["batch_size"])
    parser.add_argument(
        "--learning-rate", type=float, default=cfg.CONFIG["learning_rate"]
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = dict(cfg.CONFIG)
    config["num_epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.learning_rate

    np.random.seed(config["random_seed"])
    tf.random.set_seed(config["random_seed"])

    os.makedirs(args.model_dir, exist_ok=True)

    # 1. explore + organize data
    stats, _ = explore_dataset(args.dataset_path)
    print_dataset_summary(stats)

    organized_path, class_names = prepare_data_structure(
        args.dataset_path, args.organized_path
    )
    num_classes = len(class_names)
    print(f"\nNumber of classes to predict: {num_classes}")

    # 2. data generators
    train_gen, val_gen = create_data_generators(organized_path, config)
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")

    # 3. build + compile model
    model = build_custom_cnn(
        input_shape=(*config["img_size"], 1), num_classes=num_classes
    )
    model.summary()

    optimizer = Adam(learning_rate=config["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", "top_k_categorical_accuracy"],
    )

    best_model_path = os.path.join(args.model_dir, "best_circuit_model.h5")
    callbacks = [
        ModelCheckpoint(
            best_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy", patience=15, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
        ),
    ]

    # 4. train
    history = model.fit(
        train_gen,
        epochs=config["num_epochs"],
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
    )

    # 5. save final artifacts
    best_model = tf.keras.models.load_model(best_model_path)
    val_loss, val_acc, val_top_k = best_model.evaluate(val_gen, verbose=0)
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Top-3 Accuracy: {val_top_k:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")

    final_model_path = os.path.join(args.model_dir, "circuit_classifier.keras")
    best_model.save(final_model_path)

    metadata = {
        "class_names": class_names,
        "num_classes": num_classes,
        "input_shape": [*config["img_size"], 1],
        "model_type": cfg.MODEL_TYPE,
        "config": config,
        "performance": {
            "val_accuracy": float(val_acc),
            "val_loss": float(val_loss),
            "val_top_k_accuracy": float(val_top_k),
        },
    }
    with open(os.path.join(args.model_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Model saved successfully!")
    return history


if __name__ == "__main__":
    main()
