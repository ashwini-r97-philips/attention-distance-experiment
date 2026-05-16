"""Inspect CUB-200-2011 dataset from HuggingFace.

Prints schema, splits, column names, example records. Confirms bbox format.
Saves a dataset_schema_report.txt and sample images with bbox overlay.

Usage:
    python inspect_cub_dataset.py --hf-dataset bentrevett/caltech-ucsd-birds-200-2011 \
        --output runs/cub/cub_dataset_inspection/
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "bimodal_head_specialisation"))

import numpy as np
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Inspect CUB-200-2011 HuggingFace dataset")
    parser.add_argument("--hf-dataset", default="bentrevett/caltech-ucsd-birds-200-2011",
                        help="HuggingFace dataset ID")
    parser.add_argument("--output", default="runs/cub/cub_dataset_inspection/",
                        help="Output directory for report and samples")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("=" * 70)
    log(f"CUB-200-2011 Dataset Inspection")
    log(f"HuggingFace ID: {args.hf_dataset}")
    log("=" * 70)

    # Load dataset
    from datasets import load_dataset
    log("\nLoading dataset (will cache locally)...")
    try:
        ds = load_dataset(args.hf_dataset)
    except Exception as e:
        log(f"\n[ERROR] Failed to load dataset: {e}")
        log(f"Try a different dataset ID with --hf-dataset")
        sys.exit(1)

    # Splits
    log(f"\nSplits: {list(ds.keys())}")
    for split_name, split_ds in ds.items():
        log(f"  {split_name}: {len(split_ds)} examples")

    # Column names and types
    first_split = list(ds.keys())[0]
    features = ds[first_split].features
    log(f"\nColumns (from '{first_split}' split):")
    for col_name, col_type in features.items():
        log(f"  {col_name}: {col_type}")

    # Check for expected columns
    expected = ["image", "label", "bbox"]
    found = {col: col in features for col in expected}
    log(f"\nExpected columns check:")
    for col, present in found.items():
        log(f"  {col}: {'✓' if present else '✗ MISSING'}")

    if not found.get("image"):
        log("[ERROR] No 'image' column found. Cannot proceed.")
        sys.exit(1)

    # Number of classes
    if "label" in features:
        all_labels = ds[first_split]["label"]
        n_classes = len(set(all_labels))
        log(f"\nNumber of classes: {n_classes}")
        label_counts = Counter(all_labels)
        log(f"  Min examples per class: {min(label_counts.values())}")
        log(f"  Max examples per class: {max(label_counts.values())}")
        log(f"  Mean examples per class: {np.mean(list(label_counts.values())):.1f}")

    # Example records
    log(f"\nFirst 5 example records (from '{first_split}'):")
    for i in range(min(5, len(ds[first_split]))):
        ex = ds[first_split][i]
        log(f"\n  Example {i}:")
        log(f"    label: {ex.get('label', 'N/A')}")
        if "bbox" in ex:
            log(f"    bbox: {ex['bbox']}")
        if "image" in ex:
            img = ex["image"]
            log(f"    image size: {img.size} (W×H), mode: {img.mode}")
        # Print any other columns
        for col in features:
            if col not in ("image", "label", "bbox"):
                val = ex.get(col)
                if val is not None:
                    val_str = str(val)[:100]
                    log(f"    {col}: {val_str}")

    # Image size distribution
    log(f"\nImage size distribution (first 200 images from '{first_split}'):")
    widths, heights = [], []
    for i in range(min(200, len(ds[first_split]))):
        img = ds[first_split][i]["image"]
        widths.append(img.size[0])
        heights.append(img.size[1])
    log(f"  Width:  min={min(widths)}, max={max(widths)}, "
        f"mean={np.mean(widths):.0f}, median={np.median(widths):.0f}")
    log(f"  Height: min={min(heights)}, max={max(heights)}, "
        f"mean={np.mean(heights):.0f}, median={np.median(heights):.0f}")

    # Bbox format analysis
    if "bbox" in features:
        log(f"\nBounding box analysis (first 100 examples):")
        bbox_formats = []
        for i in range(min(100, len(ds[first_split]))):
            bbox = ds[first_split][i]["bbox"]
            if bbox is not None:
                bbox_formats.append(len(bbox) if isinstance(bbox, (list, tuple)) else "scalar")
        if bbox_formats:
            log(f"  Format: list/tuple of length {bbox_formats[0]}")
            # Check if [x0, y0, x1, y1] or [x, y, w, h]
            ex0 = ds[first_split][0]
            bbox = ex0["bbox"]
            img_w, img_h = ex0["image"].size
            log(f"  Sample bbox: {bbox}  (image size: {img_w}×{img_h})")
            if len(bbox) == 4:
                x0, y0, x1, y1 = bbox
                if x1 > img_w or y1 > img_h:
                    log(f"  [WARNING] bbox values exceed image size — might be [x,y,w,h] format")
                elif x1 > x0 and y1 > y0:
                    log(f"  Interpretation: [x0, y0, x1, y1] (top-left, bottom-right)")
                    log(f"  Box width: {x1-x0}, height: {y1-y0}")
                else:
                    log(f"  [WARNING] Unusual bbox values — check format manually")

    # Save report
    report_path = os.path.join(args.output, "dataset_schema_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    log(f"\nReport saved → {report_path}")

    # Save sample images with bbox overlay
    if "bbox" in features and "image" in features:
        log(f"\nSaving 10 sample images with bbox overlay...")
        try:
            from PIL import ImageDraw
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            samples_dir = os.path.join(args.output, "sample_images")
            os.makedirs(samples_dir, exist_ok=True)

            for i in range(min(10, len(ds[first_split]))):
                ex = ds[first_split][i]
                img = ex["image"].copy()
                bbox = ex["bbox"]
                label = ex.get("label", "?")

                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                ax.imshow(img)

                if bbox and len(bbox) == 4:
                    x0, y0, x1, y1 = bbox
                    rect = mpatches.Rectangle(
                        (x0, y0), x1 - x0, y1 - y0,
                        linewidth=2, edgecolor="lime", facecolor="none"
                    )
                    ax.add_patch(rect)

                ax.set_title(f"idx={i}, label={label}, bbox={bbox}\nimg_size={img.size}",
                             fontsize=8)
                ax.axis("off")
                plt.tight_layout()
                plt.savefig(os.path.join(samples_dir, f"sample_{i:04d}.png"),
                            dpi=100, bbox_inches="tight")
                plt.close()

            log(f"  Saved → {samples_dir}/")
        except Exception as e:
            log(f"  [WARNING] Could not save sample images: {e}")

    log(f"\n{'='*70}")
    log("Inspection complete.")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
