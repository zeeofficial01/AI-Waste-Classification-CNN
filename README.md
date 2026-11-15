# AI-Based Waste Classification System Using CNN

This project builds an image classification system to identify waste types using a Convolutional Neural Network (CNN) with TensorFlow/Keras. It includes training/evaluation scripts, saved artifacts, and a Gradio web app for image (and optional video) inference.

## Waste Categories
The model is trained on 6 classes:
- `cardboard`
- `glass`
- `metal`
- `paper`
- `plastic`
- `trash`

## Project Structure
```text
waste_classification_cnn/
├── app.py                     # Gradio app for image and video classification
├── train.py                   # Dataset loading, preprocessing, training, evaluation
├── download_dataset.py        # Helper to download Kaggle dataset
├── requirements.txt
├── README.md
├── data/
│   └── garbage_classification/  # Dataset directory (created after download)
├── models/
│   ├── waste_cnn.keras        # Saved trained model
│   └── class_names.json       # Class label mapping
└── outputs/
    ├── sample_images.png
    ├── training_history.png
    ├── confusion_matrix.png
    └── classification_report.txt
```

## 1. Setup

### Create and activate a virtual environment (recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

> TensorFlow installation may vary by OS/GPU. If you have a compatible GPU, install the appropriate TensorFlow build for acceleration.

## 2. Download the Dataset (Kaggle)

Dataset: [Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)

### Option A: Using Kaggle API (recommended)
1. Create a Kaggle API token (`kaggle.json`) from your Kaggle account settings.
2. Place it at `~/.kaggle/kaggle.json` and set permissions:
   ```bash
   mkdir -p ~/.kaggle
   cp /path/to/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. Run the download helper:
   ```bash
   python download_dataset.py
   ```

### Option B: Manual download
1. Download and extract the dataset from Kaggle.
2. Place the extracted class folders under:
   ```
   data/garbage_classification/
   ├── cardboard/
   ├── glass/
   ├── metal/
   ├── paper/
   ├── plastic/
   └── trash/
   ```

## 3. Train and Evaluate the Model

Run:
```bash
python train.py
```

This script will:
- Load and explore the dataset (class counts, sample visualization)
- Preprocess and augment images
- Split into training/validation/test sets (stratified)
- Build a CNN (Conv2D + MaxPooling + Dropout + Dense)
- Train the model (configurable epochs/batch size)
- Plot training/validation accuracy and loss
- Evaluate on the test set (accuracy, weighted F1 score, confusion matrix, classification report)
- Save the trained model and artifacts in `models/` and `outputs/`

### Useful training options
```bash
python train.py --epochs 25 --batch-size 32 --img-size 224 --learning-rate 0.001
```

## 4. Run the Gradio Web App

After training (so `models/waste_cnn.keras` and `models/class_names.json` exist), start the app:
```bash
python app.py
```

The app provides:
- **Image Classification**: Upload a waste image to get predicted class and confidence scores
- **Video Classification (Optional)**: Upload a video; the app samples 1–2 frames/sec, overlays predictions/confidence on frames, and exports a processed video

## 5. Testing and Observations
- Test with images not used during training (the held-out test set or your own photos).
- Review `outputs/confusion_matrix.png` and `outputs/classification_report.txt` to identify commonly confused classes.
- Common challenges: visually similar materials (e.g., glass vs plastic), lighting/background variation, and class imbalance (especially `trash`).

## Notes on Reproducibility
- The training script uses a fixed random seed where possible, but some GPU/non-deterministic operations may still introduce small run-to-run variation.
- You can adjust architecture, augmentation, image size, and training hyperparameters in `train.py` to experiment and improve performance.

## Optional Improvements
- Transfer learning (e.g., MobileNetV2/EfficientNet) for higher accuracy
- Class weighting or resampling to handle imbalance
- More extensive data augmentation and hyperparameter tuning
- Frame aggregation/smoothing for video predictions
