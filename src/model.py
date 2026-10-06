"""Model definitions for pneumonia classification from chest X-ray scans.

This module contains the two training choices used by the project:
1. a compact CNN built from scratch, and
2. a ResNet50 transfer-learning model for a stronger feature extractor.

Every layer is intentionally documented so the project remains easy to read,
modify, and explain during academic or portfolio work.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


IMAGE_SIZE = (224, 224)
NUM_CLASSES = 2


def build_augmentation_layer() -> tf.keras.Sequential:
    """Create light image transforms used only during training.

    These operations make the network more robust to small shifts, rotations,
    zooms, and intensity changes in X-ray scans. They are not used at
    prediction time, which keeps inference stable and deterministic.
    """
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.12),
            layers.RandomContrast(0.12),
        ],
        name="xray_augmentation",
    )


def conv_block(inputs: tf.Tensor, filters: int, dropout_rate: float) -> tf.Tensor:
    """Build one CNN block with convolution, normalization, and dropout.

    Each block follows a common pattern:
    - convolution to learn local patterns,
    - batch normalization to stabilize training,
    - ReLU activation to add non-linearity,
    - pooling to reduce spatial size,
    - dropout to reduce overfitting.
    """
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    return layers.Dropout(dropout_rate)(x)


def build_custom_cnn(input_shape: tuple[int, int, int] = (224, 224, 3)) -> tf.keras.Model:
    """Create a lightweight custom CNN for pneumonia detection.

    This is the project's custom architecture. It learns features from scratch
    while using augmentation, batch normalization, and dropout for a better
    balance of accuracy and generalization.
    """
    inputs = layers.Input(shape=input_shape)

    # Augmentation is placed inside the model so the saved training pipeline is
    # self-contained and easy to reuse for future training runs.
    x = build_augmentation_layer()(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    # The filter size grows gradually so early layers detect simple patterns and
    # later layers capture more complex disease-specific structures.
    x = conv_block(x, filters=32, dropout_rate=0.15)
    x = conv_block(x, filters=64, dropout_rate=0.20)
    x = conv_block(x, filters=128, dropout_rate=0.25)
    x = conv_block(x, filters=192, dropout_rate=0.30)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return models.Model(inputs, outputs, name="custom_pneumonia_cnn")


def build_resnet50(input_shape: tuple[int, int, int] = (224, 224, 3)) -> tf.keras.Model:
    """Create a transfer-learning model based on ResNet50.

    This approach reuses ImageNet-trained convolutional features and then adds a
    small classification head for the chest X-ray problem. It is useful when the
    dataset is limited and a more capable backbone is needed.
    """
    inputs = layers.Input(shape=input_shape)

    # Augmentation is applied before ResNet preprocessing, which keeps the raw
    # image pipeline simple while still allowing conventional ImageNet inputs.
    x = build_augmentation_layer()(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)

    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.45)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return models.Model(inputs, outputs, name="resnet50_pneumonia_classifier")


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    """Compile the model for binary-style classification over folder labels.

    The folder structure is already mapped to integer class IDs (for example,
    NORMAL=0 and PNEUMONIA=1). Sparse categorical cross-entropy matches this
    label format directly without needing one-hot encoding.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
