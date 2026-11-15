#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for the Garbage Classification dataset.

This script:
1. Loads images from data/garbage_classification/
2. Calculates the number of images in each class
3. Displays 5 random sample images from each category in a grid
4. Generates a class distribution bar chart
5. Analyzes image dimensions (width and height)
6. Checks color channels and image modes
7. Generates an image dimensions distribution chart
8. Saves all visualizations inside outputs/plots/
9. Prints a clear statistical summary

Important:
- Does NOT modify, delete, or move original dataset images
- Does NOT start CNN training
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# -----------------------------
# Configuration / constants
# -----------------------------
EXPECTED_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}  # scan common image files (read-only)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments so the script is easy to reuse."""
    parser = argparse.ArgumentParser(description="Explore the waste classification dataset (EDA).")
    parser.add_argument(
        "--data-dir",
        default="data/garbage_classification",
        help="Path to dataset root containing class folders (default: data/garbage_classification)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/plots",
        help="Directory to save EDA plots (default: outputs/plots)",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=5,
        help="Number of random sample images to show per class (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sample image selection (default: 42)",
    )
    return parser.parse_args()


def collect_image_paths(data_dir: Path) -> dict[str, list[Path]]:
    """
    Section: Load dataset file paths (read-only)
    - Scans each expected class folder under data_dir
    - Collects image file paths without modifying anything
    """
    class_to_paths: dict[str, list[Path]] = {cls: [] for cls in EXPECTED_CLASSES}

    for cls in EXPECTED_CLASSES:
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            print(f"WARNING: Missing class folder: {cls_dir}")
            continue

        # Recursively find files with image extensions (read-only scan)
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                class_to_paths[cls].append(p)

        # Sort for stable ordering (random sampling is controlled separately by seed)
        class_to_paths[cls].sort()

    return class_to_paths


def count_images_per_class(class_to_paths: dict[str, list[Path]]) -> dict[str, int]:
    """Section: Calculate number of images in each class."""
    return {cls: len(paths) for cls, paths in class_to_paths.items()}


def save_sample_image_grid(
    class_to_paths: dict[str, list[Path]],
    output_path: Path,
    samples_per_class: int = 5,
    seed: int = 42,
) -> None:
    """
    Section: Display/save 5 random sample images from each category in a grid
    - Selects random samples per class (read-only)
    - Saves a grid image to outputs/plots/
    """
    rng = random.Random(seed)
    classes = [cls for cls in EXPECTED_CLASSES if cls in class_to_paths]

    rows = len(classes)
    cols = samples_per_class
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    # Make axes indexing consistent when rows/cols == 1
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    for r, cls in enumerate(classes):
        paths = class_to_paths.get(cls, [])
        if len(paths) == 0:
            chosen = []
        elif len(paths) <= samples_per_class:
            chosen = list(paths)
            rng.shuffle(chosen)
        else:
            chosen = rng.sample(paths, samples_per_class)

        for c in range(cols):
            ax = axes[r, c]
            ax.axis("off")

            if c < len(chosen):
                img_path = chosen[c]
                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")  # convert for consistent display only (no file changes)
                        ax.imshow(img)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error\n{type(e).__name__}", ha="center", va="center", fontsize=8)
            else:
                ax.text(0.5, 0.5, "No image", ha="center", va="center", fontsize=8)

            if c == 0:
                ax.set_title(cls, fontsize=12, fontweight="bold", loc="left")

    plt.suptitle("Random Sample Images per Class", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sample image grid: {output_path}")


def save_class_distribution_bar_chart(class_counts: dict[str, int], output_path: Path) -> None:
    """
    Section: Generate a class distribution bar chart
    - Visualizes how many images each class contains
    """
    classes = list(class_counts.keys())
    counts = [class_counts[c] for c in classes]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, counts, color="#4C78A8")
    plt.title("Class Distribution (Image Counts)", fontsize=14, fontweight="bold")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=20)
    plt.grid(axis="y", alpha=0.3)

    # Add count labels on top of bars for readability
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count),
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved class distribution bar chart: {output_path}")


def analyze_images(class_to_paths: dict[str, list[Path]]) -> dict:
    """
    Section: Analyze image dimensions, modes, and color channels
    - Opens images in read-only mode to collect metadata
    - Tracks width/height, modes, and inferred channel counts
    """
    widths = []
    heights = []
    areas = []
    aspect_ratios = []
    modes = Counter()
    channels = Counter()
    mode_by_class = defaultdict(Counter)
    unreadable_files = []  # (class, path, error)

    for cls, paths in class_to_paths.items():
        for p in paths:
            try:
                with Image.open(p) as img:
                    w, h = img.size
                    mode = img.mode
                    # Infer number of channels from PIL mode where possible
                    channel_count = len(img.getbands())

                widths.append(w)
                heights.append(h)
                areas.append(w * h)
                aspect_ratios.append(w / h if h else 0)
                modes[mode] += 1
                channels[channel_count] += 1
                mode_by_class[cls][mode] += 1
            except Exception as e:
                unreadable_files.append((cls, str(p), f"{type(e).__name__}: {e}"))

    return {
        "widths": widths,
        "heights": heights,
        "areas": areas,
        "aspect_ratios": aspect_ratios,
        "modes": modes,
        "channels": channels,
        "mode_by_class": mode_by_class,
        "unreadable_files": unreadable_files,
    }


def save_dimensions_distribution_chart(analysis: dict, output_path: Path) -> None:
    """
    Section: Generate an image dimensions distribution chart
    - Creates histograms for widths and heights
    - Creates a scatter plot of width vs height to show size variety
    """
    widths = np.array(analysis["widths"], dtype=np.int32) if analysis["widths"] else np.array([])
    heights = np.array(analysis["heights"], dtype=np.int32) if analysis["heights"] else np.array([])

    if len(widths) == 0 or len(heights) == 0:
        print("WARNING: No readable images found for dimensions distribution chart.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Histogram: Widths
    axes[0].hist(widths, bins=30, color="#72B7B2", edgecolor="black", alpha=0.8)
    axes[0].set_title("Image Width Distribution")
    axes[0].set_xlabel("Width (pixels)")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(alpha=0.3)

    # Histogram: Heights
    axes[1].hist(heights, bins=30, color="#F58518", edgecolor="black", alpha=0.8)
    axes[1].set_title("Image Height Distribution")
    axes[1].set_xlabel("Height (pixels)")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(alpha=0.3)

    # Scatter: Width vs Height
    axes[2].scatter(widths, heights, alpha=0.4, s=12, color="#54A24B")
    axes[2].set_title("Width vs Height")
    axes[2].set_xlabel("Width (pixels)")
    axes[2].set_ylabel("Height (pixels)")
    axes[2].grid(alpha=0.3)

    plt.suptitle("Image Dimensions Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved image dimensions distribution chart: {output_path}")


def print_statistical_summary(class_counts: dict[str, int], analysis: dict) -> None:
    """Section: Print a clear statistical summary of the dataset."""
    widths = np.array(analysis["widths"], dtype=np.int32) if analysis["widths"] else np.array([])
    heights = np.array(analysis["heights"], dtype=np.int32) if analysis["heights"] else np.array([])
    aspect_ratios = np.array(analysis["aspect_ratios"], dtype=np.float32) if analysis["aspect_ratios"] else np.array([])
    unreadable_files = analysis["unreadable_files"]

    total_images = sum(class_counts.values())

    print("\n" + "=" * 70)
    print("DATASET EXPLORATORY ANALYSIS SUMMARY")
    print("=" * 70)

    # Class counts
    print("\n1) Class image counts")
    for cls, count in class_counts.items():
        pct = (count / total_images * 100) if total_images else 0.0
        print(f"  - {cls:<10}: {count:>5} images ({pct:5.2f}%)")
    print(f"  {'TOTAL':<10}: {total_images:>5} images")

    # Basic class balance info
    if class_counts:
        max_cls = max(class_counts, key=class_counts.get)
        min_cls = min(class_counts, key=class_counts.get)
        print(f"\n  Largest class : {max_cls} ({class_counts[max_cls]})")
        print(f"  Smallest class: {min_cls} ({class_counts[min_cls]})")
        if class_counts[min_cls] > 0:
            ratio = class_counts[max_cls] / class_counts[min_cls]
            print(f"  Imbalance ratio (max/min): {ratio:.2f}x")

    # Dimensions summary
    print("\n2) Image dimensions summary (width/height in pixels)")
    if len(widths) > 0 and len(heights) > 0:
        print(f"  Width  - min: {widths.min()}, max: {widths.max()}, mean: {widths.mean():.2f}, median: {np.median(widths):.2f}")
        print(f"  Height - min: {heights.min()}, max: {heights.max()}, mean: {heights.mean():.2f}, median: {np.median(heights):.2f}")
        print(f"  Aspect ratio - min: {aspect_ratios.min():.3f}, max: {aspect_ratios.max():.3f}, mean: {aspect_ratios.mean():.3f}")
        unique_sizes = len(set(zip(widths.tolist(), heights.tolist())))
        print(f"  Unique (width, height) combinations: {unique_sizes}")
    else:
        print("  No dimension data available (no readable images).")

    # Modes/channels summary
    print("\n3) Image modes and color channels")
    if analysis["modes"]:
        print("  Modes:")
        for mode, count in analysis["modes"].most_common():
            print(f"    - {mode}: {count}")
    else:
        print("  Modes: none detected")

    if analysis["channels"]:
        print("  Inferred channel counts:")
        for ch, count in analysis["channels"].most_common():
            print(f"    - {ch} channel(s): {count}")
    else:
        print("  Channel counts: none detected")

    print("\n  Modes by class:")
    for cls in EXPECTED_CLASSES:
        cls_modes = analysis["mode_by_class"].get(cls, Counter())
        if cls_modes:
            modes_str = ", ".join([f"{m}={n}" for m, n in cls_modes.most_common()])
            print(f"    - {cls:<10}: {modes_str}")
        else:
            print(f"    - {cls:<10}: no data")

    # Readability note during EDA scan
    print("\n4) Readability issues encountered during EDA scan")
    if unreadable_files:
        print(f"  Unreadable files: {len(unreadable_files)}")
        for cls, path, err in unreadable_files[:10]:
            print(f"    - [{cls}] {path} -> {err}")
        if len(unreadable_files) > 10:
            print(f"    ... and {len(unreadable_files) - 10} more")
    else:
        print("  Unreadable files: 0")

    print("\n" + "=" * 70)
    print("EDA complete. Original dataset was not modified. CNN training was not started.")
    print("=" * 70 + "\n")


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print("=== Exploratory Data Analysis (EDA) ===")
    print(f"Dataset directory: {data_dir.resolve()}")
    print(f"Output plots directory: {output_dir.resolve()}")
    print("Note: This script is read-only and will not modify/delete/move dataset images.\n")

    if not data_dir.exists() or not data_dir.is_dir():
        print(f"ERROR: Dataset directory not found: {data_dir}")
        return 1

    # 1) Load image paths from class folders
    class_to_paths = collect_image_paths(data_dir)

    # 2) Calculate number of images in each class
    class_counts = count_images_per_class(class_to_paths)
    print("Class counts:")
    for cls, count in class_counts.items():
        print(f"  - {cls}: {count}")
    print()

    # 3) Save grid of random sample images per class
    save_sample_image_grid(
        class_to_paths=class_to_paths,
        output_path=output_dir / "sample_images_grid.png",
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )

    # 4) Save class distribution bar chart
    save_class_distribution_bar_chart(class_counts, output_dir / "class_distribution.png")

    # 5,6) Analyze dimensions, modes, and channels
    analysis = analyze_images(class_to_paths)

    # 7) Save dimensions distribution chart
    save_dimensions_distribution_chart(analysis, output_dir / "image_dimensions_distribution.png")

    # 8 is handled by all save_* functions above (outputs/plots)
    # 9) Print statistical summary
    print_statistical_summary(class_counts, analysis)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
