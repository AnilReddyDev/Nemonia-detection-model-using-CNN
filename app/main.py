"""FastAPI backend for the pneumonia chest X-ray prediction app.

This file is the public entry point for the web app. It serves the HTML UI,
loads the trained model once at startup, and exposes an API endpoint where a
user can upload an X-ray image and receive a prediction.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from predict import load_class_names, load_trained_model, predict_image  # noqa: E402


app = FastAPI(title="Pneumonia X-ray Classifier")
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "app" / "static"), name="static")

# These variables are kept at module scope so the model is loaded once and then
# reused for every uploaded image instead of reloading it for every request.
model = None
class_names: list[str] = []
model_error: str | None = None


@app.on_event("startup")
def startup() -> None:
    """Load the model and labels once when the backend starts."""
    global model, class_names, model_error
    try:
        model = load_trained_model()
        class_names = load_class_names()
        model_error = None
    except Exception as exc:
        model_error = str(exc)


@app.get("/")
def index() -> FileResponse:
    """Serve the frontend HTML page for the single-page web app."""
    return FileResponse(PROJECT_ROOT / "app" / "static" / "index.html")


@app.get("/health")
def liveness() -> dict[str, str]:
    """Confirm the web process is responding, independently of model readiness."""
    return {"status": "ok"}


@app.get("/api/health")
def health() -> dict[str, bool | str | None]:
    """Return the model status so the frontend can tell the user what to do next."""
    return {"model_loaded": model is not None, "error": model_error}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, float | str]:
    """Accept an uploaded chest X-ray image and return the prediction."""
    if model is None:
        raise HTTPException(status_code=503, detail=model_error or "Model is not loaded.")

    try:
        # Open the uploaded image and keep the image processing logic inside the
        # backend so the frontend only handles UI actions.
        image = Image.open(file.file)
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Upload a valid image file.") from exc

    return predict_image(model, image, class_names)
