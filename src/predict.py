"""Load a trained model and run single-image inference.

This module is used by the FastAPI backend. It reads the saved model and label
names, converts one uploaded image into the format expected by Keras, and then
returns the probability and predicted class.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from model import IMAGE_SIZE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "pneumonia_model.keras"
DEFAULT_CLASS_PATH = PROJECT_ROOT / "models" / "class_names.json"


def load_class_names(class_path: Path = DEFAULT_CLASS_PATH) -> list[str]:
    """Read label names saved after training.

    If the JSON file is missing, the code falls back to the standard Kaggle
    class names so the project can still run in a simple local setup.
    """
    if class_path.exists():
        return json.loads(class_path.read_text(encoding="utf-8"))
    return ["NORMAL", "PNEUMONIA"]


def load_trained_model(model_path: Path = DEFAULT_MODEL_PATH) -> tf.keras.Model:
    """Load the trained model once and reuse it across API requests."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train first with: python src/train.py --architecture custom"
        )
    return tf.keras.models.load_model(model_path)


def image_to_batch(image: Image.Image) -> np.ndarray:
    """Convert a PIL image into a batch tensor for Keras inference.

    Keras expects shape (batch, height, width, channels). The uploaded image is
    resized to the model input size and expanded to a 4D array.
    """
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def predict_image(model: tf.keras.Model, image: Image.Image, class_names: list[str]) -> dict[str, float | str]:
    """Return predicted class and confidence for a single chest X-ray image."""
    probabilities = model.predict(image_to_batch(image), verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    return {
        "label": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
    }
