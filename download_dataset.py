#!/usr/bin/env python3
"""
Download the Kaggle Garbage Classification dataset into data/garbage_classification.

Prerequisites:
- Kaggle API credentials configured (~/.kaggle/kaggle.json)
- `kaggle` package installed (included in requirements.txt)

Dataset: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DATASET_SLUG = "asdasdasasdas/garbage-classification"
EXPECTED_CLASSES = {"cardboard", "glass", "metal", "paper", "plastic", "trash"}


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def find_class_root(extract_dir: Path) -> Path | None:
    """Find a directory whose immediate subdirectories contain the expected class folders."""
    candidates = [extract_dir] + [p for p in extract_dir.rglob("*") if p.is_dir()]
    for d in candidates:
        try:
            subdirs = {c.name for c in d.iterdir() if c.is_dir()}
        except PermissionError:
            continue
        if EXPECTED_CLASSES.issubset(subdirs):
            return d
    return None


def copy_tree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and prepare the Garbage Classification dataset.")
    parser.add_argument("--data-dir", default="data", help="Base data directory (default: data)")
    parser.add_argument("--force", action="store_true", help="Redownload even if dataset appears present")
    args = parser.parse_args()

    base_dir = Path(args.data_dir)
    raw_dir = base_dir / "_raw_garbage_classification"
    out_dir = base_dir / "garbage_classification"
    zip_path = base_dir / "garbage-classification.zip"

    if out_dir.exists() and not args.force:
        existing = {p.name for p in out_dir.iterdir() if p.is_dir()} if out_dir.is_dir() else set()
        if EXPECTED_CLASSES.issubset(existing):
            print(f"Dataset already prepared at: {out_dir}")
            print("Use --force to redownload.")
            return 0

    base_dir.mkdir(parents=True, exist_ok=True)
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Download via Kaggle CLI (uses ~/.kaggle/kaggle.json)
    try:
        run(["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", str(base_dir), "--force"])
    except FileNotFoundError:
        print("ERROR: 'kaggle' CLI not found. Install dependencies with: pip install -r requirements.txt")
        return 1
    except subprocess.CalledProcessError as e:
        print("ERROR: Kaggle download failed.")
        print("Make sure your Kaggle API token is configured at ~/.kaggle/kaggle.json")
        print(f"Details: {e}")
        return 1

    # Kaggle typically saves as garbage-classification.zip in the target path
    candidate_zips = sorted(base_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidate_zips:
        print("ERROR: No zip file found after download.")
        return 1
    zip_path = candidate_zips[0]

    print(f"Extracting {zip_path} -> {raw_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)

    class_root = find_class_root(raw_dir)
    if class_root is None:
        print("ERROR: Could not locate expected class folders after extraction.")
        print(f"Inspected: {raw_dir}")
        return 1

    print(f"Preparing dataset at: {out_dir}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    copy_tree_contents(class_root, out_dir)

    # Basic summary
    print("\nDataset prepared successfully.")
    print(f"Location: {out_dir.resolve()}")
    total = 0
    for cls in sorted(EXPECTED_CLASSES):
        cls_dir = out_dir / cls
        count = len([p for p in cls_dir.rglob('*') if p.is_file()]) if cls_dir.exists() else 0
        total += count
        print(f"  - {cls}: {count} images")
    print(f"  Total: {total} images")

    # Cleanup raw extraction (keep zip for reuse unless desired)
    shutil.rmtree(raw_dir, ignore_errors=True)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
