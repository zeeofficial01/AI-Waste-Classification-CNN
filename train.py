#!/usr/bin/env python3
"""
Preprocessing pipeline and lightweight CNN training script for waste classification.

What this script does (when you run it explicitly):
- Loads the existing dataset split from outputs/splits/dataset_split.csv (no new splitting)
- Builds tf.data pipelines to resize images to 96x96 and normalize pixels to [0, 1]
- Applies data augmentation ONLY to training images
- Builds a small custom CNN (CPU-friendly) with TensorFlow/Keras
- Trains with Adam + sparse categorical crossentropy
- Uses EarlyStopping and ModelCheckpoint (saves best model to models/)
- Saves training accuracy/loss graphs to outputs/plots/
- Evaluates on the test dataset ONLY after training
- Generates a confusion matrix and classification report

Important:
- Does NOT modify original images
- Does NOT modify outputs/splits/dataset_split.csv
- Does NOT start training automatically on import; training runs only via CLI execution
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

# -----------------------------
# Reproducibility (set seeds early)
# -----------------------------
SEED_DEFAULT = 42


def set_global_seeds(seed: int = SEED_DEFAULT) -> None:
    """Set Python/NumPy/TensorFlow seeds for more reproducible runs (best effort on CPU)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# Heavy imports are placed after seed helper definition but executed at import time.
# Training still only starts in main() when run as a script.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # noqa: E402

EXPECTED_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


# -----------------------------
# 1) Data loading from existing CSV split
# -----------------------------

def load_split_csv(split_csv_path: Path) -> tuple[dict[str, list[tuple[str, str]]], list[str]]:
    """
    Load image paths and labels from the existing split CSV.
    Expected columns: image_path, class_name, split
    Returns:
      - split_to_records: {"train": [(path,label), ...], "val": [...], "test": [...]}
      - class_names: sorted unique class names found in the CSV
    """
    if not split_csv_path.exists():
        raise FileNotFoundError(
            f"Split CSV not found: {split_csv_path}. "
            "Run split_dataset.py first to create outputs/splits/dataset_split.csv."
        )

    split_to_records: dict[str, list[tuple[str, str]]] = {"train": [], "val": [], "test": []}
    class_set = set()
    required_cols = {"image_path", "class_name", "split"}

    with open(split_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not required_cols.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Invalid split CSV format. Required columns: {sorted(required_cols)}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            img_path = (row.get("image_path") or "").strip()
            class_name = (row.get("class_name") or "").strip()
            split = (row.get("split") or "").strip().lower()

            if not img_path or not class_name or split not in split_to_records:
                continue

            split_to_records[split].append((img_path, class_name))
            class_set.add(class_name)

    class_names = sorted(class_set)
    if not class_names:
        raise ValueError(f"No valid records loaded from {split_csv_path}")
    return split_to_records, class_names


def resolve_image_path(path_str: str, project_root: Path) -> str:
    """Resolve image paths from CSV (absolute or relative to project root) into absolute paths."""
    p = Path(path_str)
    if not p.is_absolute():
        p = project_root / p
    return str(p.resolve())


def print_split_summary(split_to_records: dict[str, list[tuple[str, str]]], class_names: list[str]) -> None:
    """Print image counts per class for each subset (train/val/test)."""
    print("\nDataset split summary (from CSV):")
    for split_name in ["train", "val", "test"]:
        records = split_to_records.get(split_name, [])
        counts = Counter(label for _, label in records)
        print(f"\n  {split_name.upper()} (total: {len(records)})")
        for cls in class_names:
            print(f"    - {cls:<10}: {counts.get(cls, 0)}")


# -----------------------------
# 2) Preprocessing and tf.data pipeline
# -----------------------------

def decode_resize_normalize(path: tf.Tensor, label: tf.Tensor, img_size: tuple[int, int]):
    """
    Preprocessing for one image:
    - Read image bytes from disk
    - Decode to RGB tensor
    - Resize to img_size (e.g., 96x96)
    - Normalize pixel values to [0, 1]
    """
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def make_augmentation_layer() -> tf.keras.Sequential:
    """
    Data augmentation for TRAINING images only.
    Keeps the model lightweight and helps generalization on limited CPU resources.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="train_augmentation",
    )


def make_dataset(
    records: list[tuple[str, str]],
    class_to_index: dict[str, int],
    img_size: tuple[int, int],
    batch_size: int,
    training: bool,
    seed: int = SEED_DEFAULT,
) -> tf.data.Dataset:
    """
    Build a tf.data pipeline from CSV split records.
    - training=True  -> shuffle + augmentation applied
    - training=False -> no shuffle + no augmentation (for validation/test)
    """
    if not records:
        raise ValueError("Cannot build dataset from empty split.")

    paths = [p for p, _ in records]
    labels = [class_to_index[c] for _, c in records]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(records), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, y: decode_resize_normalize(p, y, img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training:
        aug = make_augmentation_layer()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# -----------------------------
# 3) Lightweight custom CNN model
# -----------------------------

def build_lightweight_cnn(input_shape: tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    """
    Small custom CNN for CPU-only training (no pretrained model):
    - Conv2D(16) + MaxPooling2D
    - Conv2D(32) + MaxPooling2D
    - Conv2D(64) + MaxPooling2D
    - GlobalAveragePooling2D
    - Dense
    - Dropout
    - Output Dense(6, softmax)
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same", name="conv_16")(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool_1")(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv_32")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool_2")(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv_64")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool_3")(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense_64")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="lightweight_waste_cnn")

    # Adam optimizer + sparse categorical crossentropy (labels are integer class indices)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


# -----------------------------
# 4) Training visualization and evaluation
# -----------------------------

def save_training_plots(history: tf.keras.callbacks.History, plots_dir: Path) -> Path:
    """Save training/validation accuracy and loss graphs to outputs/plots/."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    hist = history.history
    epochs = range(1, len(hist.get("loss", [])) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist.get("accuracy", []), label="Train Accuracy")
    plt.plot(epochs, hist.get("val_accuracy", []), label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist.get("loss", []), label="Train Loss")
    plt.plot(epochs, hist.get("val_loss", []), label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = plots_dir / "training_accuracy_loss.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved training plots: {out_path}")
    return out_path


def evaluate_on_test(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: list[str],
    plots_dir: Path,
    reports_dir: Path,
) -> dict:
    """
    Evaluate the FINAL (best loaded) model on the test dataset only.
    Generates confusion matrix plot and classification report.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    y_true = []
    y_pred = []

    # Iterate over test dataset (no augmentation, no shuffling)
    for batch_x, batch_y in test_ds:
        probs = model.predict(batch_x, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_pred.extend(preds.tolist())
        y_true.extend(batch_y.numpy().astype(int).tolist())

    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(class_names))))
    report_text = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print("\n=== Test Evaluation (run after training) ===")
    print(f"Test Accuracy: {acc:.4f}\n")
    print(report_text)

    # Confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    cm_path = plots_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # Classification report text file
    report_path = reports_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Test Accuracy: {acc:.6f}\n\n")
        f.write(report_text)

    print(f"Saved confusion matrix: {cm_path}")
    print(f"Saved classification report: {report_path}")

    return {
        "test_accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_path": str(cm_path),
        "classification_report_path": str(report_path),
    }


# -----------------------------
# 5) CLI and main training flow
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight waste classification CNN (CPU-friendly).")
    parser.add_argument("--split-csv", default="outputs/splits/dataset_split.csv", help="Path to split CSV")
    parser.add_argument("--img-size", type=int, default=96, help="Square image size (default: 96)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum epochs (default: 15)")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT, help="Random seed (default: 42)")
    parser.add_argument("--model-dir", default="models", help="Directory to save model artifacts (default: models)")
    parser.add_argument("--plots-dir", default="outputs/plots", help="Directory to save plots (default: outputs/plots)")
    parser.add_argument("--reports-dir", default="outputs/reports", help="Directory to save text reports (default: outputs/reports)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Safety: enforce CPU-friendly defaults requested for this milestone
    img_size = (args.img_size, args.img_size)
    batch_size = args.batch_size
    max_epochs = args.epochs

    set_global_seeds(args.seed)

    project_root = Path.cwd()
    split_csv_path = Path(args.split_csv)
    model_dir = Path(args.model_dir)
    plots_dir = Path(args.plots_dir)
    reports_dir = Path(args.reports_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=== Lightweight CNN Training (Waste Classification) ===")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Python executable: {os.sys.executable}")
    print(f"GPU devices visible to TensorFlow: {tf.config.list_physical_devices('GPU')}")
    print(f"Using split CSV: {split_csv_path}")
    print(f"Image size: {img_size}, Batch size: {batch_size}, Max epochs: {max_epochs}")

    # Load existing split (do not recreate/modify it)
    split_to_records, class_names = load_split_csv(split_csv_path)
    print(f"Classes: {class_names}")

    # Resolve image paths from CSV and verify files exist (read-only checks)
    resolved_split: dict[str, list[tuple[str, str]]] = {}
    missing_files = []
    for split_name in ["train", "val", "test"]:
        resolved_records = []
        for p_str, label in split_to_records[split_name]:
            abs_p = resolve_image_path(p_str, project_root)
            if not Path(abs_p).exists():
                missing_files.append((split_name, p_str))
            resolved_records.append((abs_p, label))
        resolved_split[split_name] = resolved_records

    if missing_files:
        print("\nERROR: Some image paths listed in the split CSV were not found on disk.")
        for split_name, p_str in missing_files[:10]:
            print(f"  - [{split_name}] {p_str}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
        return 1

    print_split_summary(resolved_split, class_names)

    # Map class names to integer indices (used with sparse categorical crossentropy)
    class_to_index = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_names)
    if num_classes != 6:
        print(f"WARNING: Expected 6 classes, found {num_classes}. Continuing with detected classes: {class_names}")

    # Build datasets (augmentation only for training)
    train_ds = make_dataset(resolved_split["train"], class_to_index, img_size, batch_size, training=True, seed=args.seed)
    val_ds = make_dataset(resolved_split["val"], class_to_index, img_size, batch_size, training=False, seed=args.seed)
    test_ds = make_dataset(resolved_split["test"], class_to_index, img_size, batch_size, training=False, seed=args.seed)

    # Build and compile lightweight CNN
    model = build_lightweight_cnn(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)
    model.summary()

    # Callbacks: save best model + stop early if validation stops improving
    best_model_path = model_dir / "best_waste_cnn.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # Train (runs only when this script is executed explicitly)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save training accuracy/loss graphs
    save_training_plots(history, plots_dir)

    # Load best saved model for final test evaluation (test evaluated only after training)
    if best_model_path.exists():
        print(f"\nLoading best saved model from: {best_model_path}")
        eval_model = tf.keras.models.load_model(best_model_path)
    else:
        print("\nWARNING: Best model checkpoint not found; evaluating current in-memory model.")
        eval_model = model

    # Save class names for app/inference reuse
    class_names_path = model_dir / "class_names.json"
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump({"class_names": class_names}, f, indent=2)
    print(f"Saved class names: {class_names_path}")

    # Evaluate on test set only after training is complete
    metrics = evaluate_on_test(eval_model, test_ds, class_names, plots_dir, reports_dir)

    # Save a small run summary
    summary_path = reports_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split_csv": str(split_csv_path),
                "img_size": img_size[0],
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "seed": args.seed,
                "classes": class_names,
                "train_count": len(resolved_split["train"]),
                "val_count": len(resolved_split["val"]),
                "test_count": len(resolved_split["test"]),
                "best_model_path": str(best_model_path),
                "test_accuracy": metrics["test_accuracy"],
            },
            f,
            indent=2,
        )
    print(f"Saved training summary: {summary_path}")
    print("\nTraining and evaluation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
