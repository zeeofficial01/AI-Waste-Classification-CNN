#!/usr/bin/env python3
"""
Verify the Garbage Classification dataset without modifying or deleting any files.

Checks performed:
- Verify all six expected class folders are present
- Count images in each category
- Check supported image formats (JPG, JPEG, PNG)
- Open every image using PIL to verify readability
- Identify corrupted/unreadable images
- Display image dimensions for sample images
- Generate a class distribution summary

This script does NOT modify/delete images and does NOT train a model.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

EXPECTED_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def iter_files(directory: Path) -> Iterable[Path]:
    """Yield files in a directory recursively (read-only scan)."""
    if not directory.exists():
        return []
    return (p for p in directory.rglob("*") if p.is_file())


def verify_with_pil(image_path: Path) -> tuple[bool, str | None, tuple[int, int] | None]:
    """Return (is_readable, error_message, dimensions). Uses PIL verify + reopen for load check."""
    try:
        # verify() checks file integrity but leaves the file object unusable for some ops
        with Image.open(image_path) as img:
            img.verify()

        # Reopen to ensure the image data can be read/decoded
        with Image.open(image_path) as img:
            img.load()
            dims = img.size  # (width, height)
        return True, None, dims
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError) as e:
        return False, f"{type(e).__name__}: {e}", None
    except Exception as e:  # Defensive catch for unexpected PIL issues
        return False, f"UnexpectedError({type(e).__name__}): {e}", None


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify dataset structure and image readability (read-only).")
    parser.add_argument(
        "--data-dir",
        default="data/garbage_classification",
        help="Path to dataset root containing class folders (default: data/garbage_classification)",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=3,
        help="Number of sample readable images per class to display dimensions for (default: 3)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print("=== Dataset Verification (Read-only) ===")
    print(f"Dataset path: {data_dir.resolve()}\n")

    if not data_dir.exists() or not data_dir.is_dir():
        print(f"ERROR: Dataset directory not found or not a directory: {data_dir}")
        return 1

    # 1) Verify all six class folders
    print("1) Class folder verification")
    existing_dirs = {p.name for p in data_dir.iterdir() if p.is_dir()}
    missing_classes = [c for c in EXPECTED_CLASSES if c not in existing_dirs]
    extra_dirs = sorted([d for d in existing_dirs if d not in EXPECTED_CLASSES])

    for cls in EXPECTED_CLASSES:
        status = "FOUND" if cls in existing_dirs else "MISSING"
        print(f"  - {cls:<10}: {status}")
    if extra_dirs:
        print(f"  Extra folders (ignored in expected summary): {extra_dirs}")
    if missing_classes:
        print(f"\nWARNING: Missing expected class folders: {missing_classes}\n")
    else:
        print("\nAll six expected class folders are present.\n")

    # 2-5) Count, format check, readability check
    print("2-5) Scanning files, checking formats, and verifying readability with PIL")
    print("(No files will be modified or deleted)\n")

    per_class_total_files = defaultdict(int)
    per_class_supported_files = defaultdict(int)
    per_class_readable_images = defaultdict(int)
    per_class_unsupported_files = defaultdict(int)
    per_class_corrupted_images = defaultdict(int)
    per_class_sample_dims = defaultdict(list)

    unsupported_files = []  # (class, path, ext)
    corrupted_or_unreadable = []  # (class, path, reason)
    format_counts = defaultdict(int)

    # Scan only expected class folders for the main report (but do not alter anything)
    for cls in EXPECTED_CLASSES:
        cls_dir = data_dir / cls
        if not cls_dir.exists() or not cls_dir.is_dir():
            continue

        for path in iter_files(cls_dir):
            per_class_total_files[cls] += 1
            ext = path.suffix.lower()
            format_counts[ext if ext else "<no_ext>"] += 1

            if ext not in SUPPORTED_EXTENSIONS:
                per_class_unsupported_files[cls] += 1
                unsupported_files.append((cls, path, ext))
                continue

            per_class_supported_files[cls] += 1
            ok, err, dims = verify_with_pil(path)
            if ok:
                per_class_readable_images[cls] += 1
                if len(per_class_sample_dims[cls]) < max(args.samples_per_class, 0):
                    per_class_sample_dims[cls].append((path, dims))
            else:
                per_class_corrupted_images[cls] += 1
                corrupted_or_unreadable.append((cls, path, err or "Unknown error"))

    # 6) Display image dimensions for sample images
    print("6) Sample image dimensions (readable images)")
    any_samples = False
    for cls in EXPECTED_CLASSES:
        samples = per_class_sample_dims.get(cls, [])
        if not samples:
            continue
        any_samples = True
        print(f"  [{cls}]")
        for path, dims in samples:
            rel = path.relative_to(data_dir) if path.is_relative_to(data_dir) else path
            if dims is None:
                print(f"    - {rel}: dimensions unavailable")
            else:
                w, h = dims
                print(f"    - {rel}: {w}x{h} (WxH)")
    if not any_samples:
        print("  No readable supported sample images found.")
    print()

    # 7) Class distribution summary
    print("7) Class distribution summary")
    header = f"{'Class':<10} {'Total':>7} {'Supported(JPG/JPEG/PNG)':>24} {'Readable':>10} {'Corrupted':>11} {'Unsupported':>13}"
    print(header)
    print("-" * len(header))

    total_files = 0
    total_supported = 0
    total_readable = 0
    total_corrupted = 0
    total_unsupported = 0

    for cls in EXPECTED_CLASSES:
        t = per_class_total_files.get(cls, 0)
        s = per_class_supported_files.get(cls, 0)
        r = per_class_readable_images.get(cls, 0)
        c = per_class_corrupted_images.get(cls, 0)
        u = per_class_unsupported_files.get(cls, 0)
        total_files += t
        total_supported += s
        total_readable += r
        total_corrupted += c
        total_unsupported += u
        pct = (r / total_readable * 100) if total_readable > 0 else 0.0
        print(f"{cls:<10} {t:>7} {s:>24} {r:>10} {c:>11} {u:>13}   ({pct:5.1f}% of readable)")

    print("-" * len(header))
    print(f"{'TOTAL':<10} {total_files:>7} {total_supported:>24} {total_readable:>10} {total_corrupted:>11} {total_unsupported:>13}")

    # Format breakdown
    print("\nObserved file extensions (all files scanned in expected class folders):")
    if format_counts:
        for ext, count in sorted(format_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            label = ext if ext else "<no_ext>"
            supported_note = "supported" if ext in SUPPORTED_EXTENSIONS else "unsupported"
            print(f"  - {label}: {count} ({supported_note})")
    else:
        print("  No files found.")

    # Corrupted/unreadable details
    print("\nCorrupted or unreadable supported images:")
    if corrupted_or_unreadable:
        for cls, path, reason in corrupted_or_unreadable:
            rel = path.relative_to(data_dir) if path.is_relative_to(data_dir) else path
            print(f"  - [{cls}] {rel} -> {reason}")
    else:
        print("  None found.")

    # Unsupported details (can be many; show summary + first few)
    print("\nUnsupported-format files (not JPG/JPEG/PNG):")
    if unsupported_files:
        print(f"  Count: {len(unsupported_files)}")
        preview = unsupported_files[:20]
        for cls, path, ext in preview:
            rel = path.relative_to(data_dir) if path.is_relative_to(data_dir) else path
            print(f"  - [{cls}] {rel} (ext: {ext or '<no_ext>'})")
        if len(unsupported_files) > len(preview):
            print(f"  ... and {len(unsupported_files) - len(preview)} more")
    else:
        print("  None found.")

    print("\nVerification complete. No files were modified or deleted. Model training was not started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
