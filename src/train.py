"""Train the chest X-ray pneumonia classifier.

This script is responsible for reading the dataset, creating the selected model,
training it with augmentation and regularization, and saving the final model
artifact plus the label names.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import tensorflow as tf

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model import IMAGE_SIZE, build_custom_cnn, build_resnet50, compile_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "chest_xray"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "pneumonia_model.keras"
DEFAULT_CLASS_PATH = PROJECT_ROOT / "models" / "class_names.json"


def load_datasets(data_dir: Path, batch_size: int) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Load the train and validation folders as TensorFlow datasets.

    The Kaggle chest X-ray dataset is already split by folder names. TensorFlow
    can read those directories directly and assign labels based on the folder
    names. If the validation folder is missing or empty, the code falls back to
    an internal split from the training directory.
    """
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists():
        raise FileNotFoundError(f"Training folder not found: {train_dir}")

    # This dataset object contains the class names metadata before we add
    # prefetching, because the prefetch wrapper does not preserve `.class_names`.
    train_source = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="int",
        shuffle=True,
    )
    class_names = train_source.class_names

    if val_dir.exists() and any(val_dir.iterdir()):
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            image_size=IMAGE_SIZE,
            batch_size=batch_size,
            label_mode="int",
            shuffle=False,
        )
    else:
        # Some Kaggle copies do not provide a proper validation folder. This
        # ensures a usable validation split still exists for monitoring.
        train_source = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=IMAGE_SIZE,
            batch_size=batch_size,
            label_mode="int",
            validation_split=0.20,
            subset="training",
            seed=42,
            shuffle=True,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=IMAGE_SIZE,
            batch_size=batch_size,
            label_mode="int",
            validation_split=0.20,
            subset="validation",
            seed=42,
            shuffle=False,
        )
        class_names = train_source.class_names

    autotune = tf.data.AUTOTUNE
    return train_source.prefetch(autotune), val_ds.prefetch(autotune), class_names


def build_training_callbacks(model_path: Path) -> list[tf.keras.callbacks.Callback]:
    """Create early stopping and checkpoint callbacks for stable training."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]


def train(args: argparse.Namespace) -> None:
    """Run the full training workflow for one selected architecture."""
    data_dir = Path(args.data_dir)
    model_path = Path(args.model_path)
    class_path = Path(args.class_path)

    # Step 1: load data and discover label names from the folder structure.
    train_ds, val_ds, class_names = load_datasets(data_dir=data_dir, batch_size=args.batch_size)

    # Step 2: select the architecture requested by the user.
    if args.architecture == "resnet50":
        model = build_resnet50()
        learning_rate = args.learning_rate or 1e-4
    else:
        model = build_custom_cnn()
        learning_rate = args.learning_rate or 1e-3

    compile_model(model, learning_rate=learning_rate)

    # Step 3: training helpers to prevent overfitting and preserve the best model.
    callbacks = build_training_callbacks(model_path)

    # Step 4: fit the model and monitor validation quality.
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    # Step 5: save the trained model and the class label list for API inference.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    class_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    class_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    print(f"Saved model to {model_path}")
    print(f"Saved class names to {class_path}: {class_names}")


def parse_args() -> argparse.Namespace:
    """Read command-line arguments that control dataset location and training."""
    parser = argparse.ArgumentParser(description="Train pneumonia X-ray classifier.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Path to chest_xray dataset.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Where to save the trained model.")
    parser.add_argument("--class-path", default=str(DEFAULT_CLASS_PATH), help="Where to save class names.")
    parser.add_argument("--architecture", choices=["custom", "resnet50"], default="custom")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
