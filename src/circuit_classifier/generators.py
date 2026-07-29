"""Data generator factory for training/validation."""

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_data_generators(data_path, config):
    """Create training and validation generators with augmentation.

    Training data is lightly augmented (small rotation/shift/shear/zoom);
    validation data is only rescaled.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.1,
        fill_mode="constant",
        cval=0,
        validation_split=config["validation_split"],
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=config["validation_split"],
    )

    train_generator = train_datagen.flow_from_directory(
        data_path,
        target_size=config["img_size"],
        batch_size=config["batch_size"],
        class_mode="categorical",
        color_mode="grayscale",
        subset="training",
        seed=config["random_seed"],
    )

    val_generator = val_datagen.flow_from_directory(
        data_path,
        target_size=config["img_size"],
        batch_size=config["batch_size"],
        class_mode="categorical",
        color_mode="grayscale",
        subset="validation",
        seed=config["random_seed"],
    )

    return train_generator, val_generator


def create_eval_generator(data_path, config):
    """Create a validation generator for evaluation/confusion-matrix use.

    Important: shuffle=False here. The training validation generator
    (from create_data_generators) shuffles by default, which is fine for
    monitoring metrics during training but makes ``val_gen.classes``
    (the true labels) fall out of sync with ``model.predict(val_gen)``
    (the predictions) — silently producing a nonsensical confusion
    matrix/classification report even though the model itself may be
    accurate. Use this generator whenever you need predictions lined up
    with ground-truth labels index-for-index.
    """
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=config["validation_split"],
    )

    return val_datagen.flow_from_directory(
        data_path,
        target_size=config["img_size"],
        batch_size=config["batch_size"],
        class_mode="categorical",
        color_mode="grayscale",
        subset="validation",
        seed=config["random_seed"],
        shuffle=False,
    )
