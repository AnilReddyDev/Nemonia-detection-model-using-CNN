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

## Important points to remember when explaining this project

- This project is a binary chest X-ray classifier: it predicts whether an image is `NORMAL` or `PNEUMONIA`.
- The model is trained on the `data/chest_xray/train` folder, which contains 5,216 images in total: 1,341 normal and 3,875 pneumonia cases.
- The validation folder `data/chest_xray/val` is used during training to monitor performance and trigger EarlyStopping, not to train the model weights.
- The test folder `data/chest_xray/test` is used for the final evaluation after training is complete.
- An epoch means one full pass through the full training dataset. In this project, with a batch size of 32, one epoch is roughly 163 batches because 5216 / 32 ≈ 163.
- The default training setting is `--epochs 10`, which means the model sees the training data 10 times unless stopped earlier by EarlyStopping.
- The training code uses validation loss to save the best model and prevent overfitting. This is visible in `src/train.py` where `EarlyStopping` and `ModelCheckpoint` are configured.
- The app is a simple end-to-end ML project: dataset -> model training -> saved model -> FastAPI backend -> browser UI.
- If you are explaining it to a reviewer, emphasize that the task is classification of lung X-rays, not generic image generation or segmentation.
- Dropout regularization: randomly turns off some neurons during training so the model does not rely too much on a few features and becomes more robust.
- Batch normalization: normalizes the activations inside each mini-batch so training becomes more stable and faster.
- Data augmentation: creates modified versions of images such as flips, zoom, rotation, and shifts so the model learns to generalize better and not memorize the training set.

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
