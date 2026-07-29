"""CNN architecture for circuit component classification."""

from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2


def build_custom_cnn(input_shape, num_classes):
    """Custom CNN optimized for hand-drawn circuit diagrams.

    Three convolutional blocks (32 -> 64 -> 128 filters) with batch
    normalization and dropout, followed by two dense layers with L2
    regularization and a softmax output.
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            # block 1
            layers.Conv2D(32, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv2D(32, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # block 2
            layers.Conv2D(64, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv2D(64, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # block 3
            layers.Conv2D(128, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv2D(128, (3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # dense head
            layers.Flatten(),
            layers.Dense(256, kernel_regularizer=l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(0.5),
            layers.Dense(128, kernel_regularizer=l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(0.5),
            # output
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model
