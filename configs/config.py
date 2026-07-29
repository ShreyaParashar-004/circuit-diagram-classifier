"""
Central configuration for the circuit component classifier.

Paths default to a local ./data layout. Override any of these via
environment variables or CLI flags in the train/predict scripts.
"""

import os

# --- Paths -------------------------------------------------------------
RAW_DATASET_PATH = os.environ.get(
    "CIRCUIT_DATASET_PATH", "./data/raw/dataset_final"
)
ORGANIZED_DATA_PATH = os.environ.get(
    "CIRCUIT_ORGANIZED_PATH", "./data/processed/organized_data"
)
MODEL_DIR = os.environ.get("CIRCUIT_MODEL_DIR", "./models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_circuit_model.h5")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "circuit_classifier.keras")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# --- Training hyperparameters -------------------------------------------
CONFIG = {
    "img_size": (64, 64),
    "batch_size": 32,
    "num_epochs": 50,
    "learning_rate": 0.001,
    "validation_split": 0.2,
    "test_split": 0.1,
    "random_seed": 42,
    "augment_data": True,
}

MODEL_TYPE = "custom_cnn"
