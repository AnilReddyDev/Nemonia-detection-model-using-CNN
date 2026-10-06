# Pneumonia X-ray Classification App

This project implements the assignment in `problemstatement.txt`: a pneumonia positive/negative classifier for chest X-ray images, with a FastAPI backend and HTML/CSS/JavaScript frontend.

The problem statement also mentions "corn leaf blight detection", but the dataset and model request are for chest X-ray pneumonia detection, so this implementation focuses on pneumonia classification.

## Project Structure

- `src/download_dataset.py` downloads the Kaggle chest X-ray pneumonia dataset when Kaggle credentials are configured.
- `src/model.py` contains a custom CNN architecture and a ResNet50 transfer-learning option.
- `src/train.py` trains the model with data augmentation, batch normalization, and dropout regularization.
- `src/predict.py` loads the trained model and predicts a single image.
- `app/main.py` serves the FastAPI API and frontend.
- `app/static/` contains the HTML, CSS, and JavaScript UI.
- `main.ipynb` mirrors the workflow for Jupyter/conda.

## Setup

Use Python 3.11 or 3.12 for TensorFlow compatibility.

```bash
conda create -n pneumonia-cnn python=3.11
conda activate pneumonia-cnn
pip install -r requirements.txt
```

## Dataset

The project expects the chest X-ray dataset to already exist locally in the following structure:

```text
data/chest_xray/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

If the dataset is not present, download it manually from the Kaggle page and extract it into the `data/` folder before training.

## Train

Custom CNN:

```bash
python src/train.py --architecture custom --epochs 10
```

ResNet50 transfer learning:

```bash
python src/train.py --architecture resnet50 --epochs 10
```

The trained model is saved to `models/pneumonia_model.keras`.

## Run App

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Deploy

For Render, set the start command to:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Upload or build `models/pneumonia_model.keras` before deployment. Large TensorFlow models may require a paid instance or a slimmer model artifact.
