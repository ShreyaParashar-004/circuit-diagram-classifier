"""
Basic sanity tests: the model builds with the expected output shape,
and the preprocessing functions return correctly-shaped, normalized
arrays. These don't require the dataset to be present.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from circuit_classifier.model import build_custom_cnn


def test_model_output_shape():
    num_classes = 12
    model = build_custom_cnn(input_shape=(64, 64, 1), num_classes=num_classes)

    dummy_input = np.random.rand(2, 64, 64, 1).astype("float32")
    output = model.predict(dummy_input, verbose=0)

    assert output.shape == (2, num_classes)
    # softmax outputs should sum to ~1 per sample
    assert np.allclose(output.sum(axis=1), 1.0, atol=1e-4)


def test_preprocess_circuit_image_normalizes(tmp_path):
    import cv2

    from circuit_classifier.preprocessing import preprocess_circuit_image

    img_path = tmp_path / "sample.png"
    dummy_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    cv2.imwrite(str(img_path), dummy_img)

    result = preprocess_circuit_image(str(img_path), target_size=(64, 64))

    assert result.shape == (64, 64)
    assert result.dtype == np.float32
    assert result.min() >= 0.0 and result.max() <= 1.0


def test_preprocess_circuit_image_missing_file_returns_none():
    from circuit_classifier.preprocessing import preprocess_circuit_image

    assert preprocess_circuit_image("/nonexistent/path.png") is None
