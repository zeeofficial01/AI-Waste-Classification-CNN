#!/usr/bin/env python3
"""
Create a reproducible, stratified train/validation/test split for the waste dataset.

Requirements implemented:
- 70% training, 15% validation, 15% testing split
- Stratified splitting to preserve class distribution
- Fixed random seed (42) for reproducibility
- No data leakage between subsets (each image path appears in exactly one split)
- Original dataset images are NOT modified, deleted, or moved
- Saves split information to a CSV file (image_path, class_name, split)
- Prints number of images per class in each subset
- Verifies no image appears in more than one subset
- Does NOT start CNN training

Note on "preprocessing" at this stage:
- This script focuses on dataset organization/splitting metadata.
- Image resizing/normalization/augmentation are handled later during model training
  (e.g., in train.py) so the original files remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# -----------------------------
# Configuration
# -----------------------------
EXPECTED_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}  # verified dataset formats


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create stratified dataset split (70/15/15) and save CSV metadata.")
    parser.add_argument(
        "--data-dir",
        default="data/garbage_classification",
        help="Path to dataset root containing class folders (default: data/garbage_classification)",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/splits/dataset_split.csv",
        help="Path to save split CSV (default: outputs/splits/dataset_split.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting (default: 42)",
    )
    return parser.parse_args()


def collect_dataset_records(data_dir: Path) -> list[dict[str, str]]:
    """
    Section 1: Load image file paths and class labels (read-only)
    - Scans each expected class folder
    - Collects records with image path and class name
    - Does not modify or move any files
    """
    records: list[dict[str, str]] = []
    missing_classes = []

    for class_name in EXPECTED_CLASSES:
        class_dir = data_dir / class_name
        if not class_dir.exists() or not class_dir.is_dir():
            missing_classes.append(class_name)
            continue

        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                # Store path as POSIX-style relative path for portability/readability
                try:
                    rel_path = p.relative_to(Path.cwd())
                except ValueError:
                    rel_path = p
                records.append({"image_path": str(rel_path).replace("\\", "/"), "class_name": class_name})

    if missing_classes:
        raise FileNotFoundError(f"Missing expected class folder(s): {missing_classes}")

    if not records:
        raise ValueError(f"No supported image files found in {data_dir}")

    # Sort for deterministic order before shuffling/splitting operations
    records.sort(key=lambda r: r["image_path"])
    return records


def print_class_counts(title: str, rows: list[dict[str, str]]) -> dict[str, int]:
    """Print and return class counts for a subset."""
    counts = Counter(row["class_name"] for row in rows)
    print(f"\n{title} (total: {len(rows)})")
    for cls in EXPECTED_CLASSES:
        print(f"  - {cls:<10}: {counts.get(cls, 0)}")
    return {cls: counts.get(cls, 0) for cls in EXPECTED_CLASSES}


def verify_no_leakage(train_rows, val_rows, test_rows) -> None:
    """
    Section 4: Verify no image appears in more than one subset
    - Uses image_path as the unique identifier
    - Raises an error if any overlap is found
    """
    train_set = set(r["image_path"] for r in train_rows)
    val_set = set(r["image_path"] for r in val_rows)
    test_set = set(r["image_path"] for r in test_rows)

    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    print("\nLeakage check (image_path overlaps):")
    print(f"  - train ∩ val : {len(train_val_overlap)}")
    print(f"  - train ∩ test: {len(train_test_overlap)}")
    print(f"  - val ∩ test  : {len(val_test_overlap)}")

    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise RuntimeError("Data leakage detected: at least one image appears in more than one subset.")

    # Extra sanity check: total unique images equals sum of subset sizes
    total_unique = len(train_set | val_set | test_set)
    total_rows = len(train_rows) + len(val_rows) + len(test_rows)
    print(f"  - unique images across all splits: {total_unique}")
    print(f"  - total assigned rows: {total_rows}")
    if total_unique != total_rows:
        raise RuntimeError("Inconsistent split assignment: duplicate rows detected within/between subsets.")

    print("  Result: PASS (no image appears in more than one subset)")


def save_split_csv(rows: list[dict[str, str]], output_csv: Path) -> None:
    """
    Section 5: Save split metadata to CSV
    Columns: image_path, class_name, split
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "class_name", "split"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved split CSV: {output_csv.resolve()}")


def main() -> int:
    args = parse_args()

    # Fixed seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    output_csv = Path(args.output_csv)

    print("=== Dataset Preprocessing & Splitting (Metadata Only) ===")
    print(f"Dataset directory: {data_dir.resolve()}")
    print(f"Random seed: {args.seed}")
    print("Split ratio: 70% train, 15% validation, 15% test (stratified)")
    print("Note: Original images will not be modified, deleted, or moved.\n")

    if not data_dir.exists() or not data_dir.is_dir():
        print(f"ERROR: Dataset directory not found: {data_dir}")
        return 1

    # 1) Load dataset records (image path + class name)
    records = collect_dataset_records(data_dir)
    image_paths = [r["image_path"] for r in records]
    labels = [r["class_name"] for r in records]

    overall_counts = Counter(labels)
    print(f"Total images found: {len(records)}")
    print("Overall class counts:")
    for cls in EXPECTED_CLASSES:
        print(f"  - {cls:<10}: {overall_counts.get(cls, 0)}")

    # 2) Stratified split: first 70% train, 30% temp
    #    Using stratify=labels preserves class proportions in train and temp sets.
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.30,          # 30% goes to temp (val + test)
        stratify=labels,         # preserve class distribution
        random_state=args.seed,  # reproducible
        shuffle=True,
    )

    # 3) Stratified split: split temp (30%) into 15% val and 15% test
    #    test_size=0.5 on the 30% temp portion gives 15%/15% overall.
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.50,
        stratify=temp_labels,    # preserve class distribution within temp split
        random_state=args.seed,
        shuffle=True,
    )

    # Build row dictionaries for each subset
    train_rows = [{"image_path": p, "class_name": y, "split": "train"} for p, y in zip(train_paths, train_labels)]
    val_rows = [{"image_path": p, "class_name": y, "split": "val"} for p, y in zip(val_paths, val_labels)]
    test_rows = [{"image_path": p, "class_name": y, "split": "test"} for p, y in zip(test_paths, test_labels)]

    # 4) Print per-class counts in each subset
    print_class_counts("Training subset class counts", train_rows)
    print_class_counts("Validation subset class counts", val_rows)
    print_class_counts("Test subset class counts", test_rows)

    # 5) Verify no leakage (no image appears in more than one subset)
    verify_no_leakage(train_rows, val_rows, test_rows)

    # 6) Save combined split metadata to CSV
    all_rows = train_rows + val_rows + test_rows
    # Sort for a clean, readable CSV (by split then class then path)
    split_order = {"train": 0, "val": 1, "test": 2}
    all_rows.sort(key=lambda r: (split_order[r["split"]], r["class_name"], r["image_path"]))
    save_split_csv(all_rows, output_csv)

    print("\nDone. CNN training was not started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
