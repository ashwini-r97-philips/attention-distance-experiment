"""ImageNet Localization Evaluation for ViT-S checkpoints.

Evaluates classification-trained checkpoints on object localization using
attention maps (last-layer CLS attention and attention rollout), without
any retraining or use of bounding-box annotations during the forward pass.

Annotation file expected at:
  <annotation_xml_dir>/<ILSVRC_id>/
    e.g.  data/imagenet_loc_annotations/ILSVRC2012_val_00000001.xml

Each XML follows the ILSVRC VOC format:
  <annotation>
    <object>
      <name>...</name>
      <bndbox><xmin>…</xmin><ymin>…</ymin><xmax>…</xmax><ymax>…</ymax></bndbox>
    </object>
  </annotation>

Usage
-----
  python eval_imagenet_localization.py \
      --checkpoints baseline:runs/baseline_vit_s16_imagenet1k/checkpoints/best.pth \
                     bimodal_high:runs/bimodal_high_vit_s16_imagenet1k/checkpoints/best.pth \
                     spread_weak:runs/spread_weak_vit_s16_imagenet1k/checkpoints/best.pth \
      --annotation-dir data/imagenet_loc_annotations \
      --num-images 2000 \
      --output-dir runs/localization_eval \
      --attention-method both
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "bimodal_head_specialisation"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label as scipy_label
from tqdm import tqdm

import timm
from datasets import load_dataset
from torchvision import transforms

from common.attention_hooks import capture_attention


# ─── Constants ────────────────────────────────────────────────────────────────

IMG_SIZE = 224
PATCH_SIZE = 16
GRID = IMG_SIZE // PATCH_SIZE  # 14
NUM_PATCHES = GRID * GRID       # 196
NUM_LAYERS = 12

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

THRESHOLDS = [0.3, 0.4, 0.5, 0.6]

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ImageNet val filenames are zero-padded 8-digit numbers
# ILSVRC2012_val_00000001.JPEG -> index 0
_FNAME_TEMPLATE = "ILSVRC2012_val_{:08d}"


# ─── Annotation loading ───────────────────────────────────────────────────────

def _parse_xml_annotation(xml_path: str) -> List[Tuple[int, int, int, int]]:
    """Parse a single ILSVRC VOC XML file and return list of (x1,y1,x2,y2) boxes.

    Coordinates are 1-indexed in the XML; we convert to 0-indexed pixels.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Original image size in the XML (may differ from 224x224)
    size_elem = root.find("size")
    orig_w = int(size_elem.find("width").text)
    orig_h = int(size_elem.find("height").text)

    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        x1 = int(float(bb.find("xmin").text)) - 1
        y1 = int(float(bb.find("ymin").text)) - 1
        x2 = int(float(bb.find("xmax").text)) - 1
        y2 = int(float(bb.find("ymax").text)) - 1

        # Scale to 224x224 (matching the CenterCrop resize pipeline)
        # First replicate Resize(256) then CenterCrop(224) on bbox coords
        scale = 256.0 / min(orig_w, orig_h)
        new_w = round(orig_w * scale)
        new_h = round(orig_h * scale)
        cx_offset = (new_w - 224) // 2
        cy_offset = (new_h - 224) // 2

        x1s = max(0, round(x1 * scale) - cx_offset)
        y1s = max(0, round(y1 * scale) - cy_offset)
        x2s = min(223, round(x2 * scale) - cx_offset)
        y2s = min(223, round(y2 * scale) - cy_offset)

        if x2s > x1s and y2s > y1s:
            boxes.append((x1s, y1s, x2s, y2s))

    return boxes


def load_annotations(annotation_dir: str) -> Dict[int, List[Tuple[int, int, int, int]]]:
    """Load all available XML annotation files from annotation_dir.

    Returns:
        dict mapping val_index (0-based) -> list of (x1,y1,x2,y2) boxes
    """
    ann_dir = Path(annotation_dir)
    if not ann_dir.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Annotation directory not found: {annotation_dir}\n"
            "Expected ILSVRC VOC-format XML files, one per validation image.\n"
            "Directory structure:\n"
            "  <annotation_dir>/ILSVRC2012_val_00000001.xml\n"
            "  <annotation_dir>/ILSVRC2012_val_00000002.xml\n"
            "  ...\n"
            "\n"
            "You can also organise them in per-class subdirs:\n"
            "  <annotation_dir>/<synset_id>/ILSVRC2012_val_00000001.xml\n"
            "\n"
            "Download from the ILSVRC 2012 devkit:\n"
            "  https://image-net.org/challenges/LSVRC/2012/index.php\n"
            "  File: ILSVRC2012_bbox_val_v3.tgz\n"
        )

    annotations: Dict[int, List[Tuple[int, int, int, int]]] = {}

    # Support flat and one-level-deep layouts
    xml_files = list(ann_dir.glob("*.xml")) + list(ann_dir.glob("*/*.xml"))

    if len(xml_files) == 0:
        raise FileNotFoundError(
            f"[ERROR] No XML files found in {annotation_dir}.\n"
            "Expected files like ILSVRC2012_val_00000001.xml"
        )

    for xml_path in xml_files:
        stem = xml_path.stem  # e.g. ILSVRC2012_val_00000001
        parts = stem.split("_")
        if len(parts) >= 3 and parts[-1].isdigit():
            idx = int(parts[-1]) - 1  # convert to 0-based
            try:
                boxes = _parse_xml_annotation(str(xml_path))
                if boxes:
                    annotations[idx] = boxes
            except Exception:
                pass  # skip malformed XML

    print(f"[INFO] Loaded annotations for {len(annotations):,} validation images.")
    return annotations


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=False,
        num_classes=1000,
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


# ─── Attention map extraction ─────────────────────────────────────────────────

def _cls_attn_last_layer(attn_dict: dict) -> np.ndarray:
    """Extract class-token attention from the last transformer block.

    attn_dict: {block_idx: (B, H, N+1, N+1)} tensors  (N = 196, +1 for CLS)
    Returns: (GRID, GRID) float32 numpy array, values in [0,1]
    """
    last_idx = max(attn_dict.keys())
    attn = attn_dict[last_idx]  # (B, H, N+1, N+1)
    # Average over heads; take CLS row (index 0), drop CLS column (index 0)
    cls_attn = attn[0].mean(0)[0, 1:]  # (N,)
    assert cls_attn.shape[0] == NUM_PATCHES, (
        f"Expected {NUM_PATCHES} patches, got {cls_attn.shape[0]}"
    )
    grid = cls_attn.reshape(GRID, GRID).cpu().float().numpy()
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    return grid


def _attention_rollout(attn_dict: dict) -> np.ndarray:
    """Compute attention rollout across all layers.

    Follows Abnar & Zuidema (2020): iteratively multiply attention matrices
    with residual identity connections.

    Returns: (GRID, GRID) float32, values in [0,1]
    """
    n_tokens = NUM_PATCHES + 1  # +1 for CLS token

    # Start with identity
    rollout = np.eye(n_tokens, dtype=np.float32)

    for layer_idx in sorted(attn_dict.keys()):
        attn = attn_dict[layer_idx]  # (B, H, N+1, N+1)
        # Mean over heads
        a = attn[0].mean(0).cpu().float().numpy()  # (N+1, N+1)
        # Add residual (identity), re-normalise rows
        a = a + np.eye(n_tokens, dtype=np.float32)
        a = a / (a.sum(axis=-1, keepdims=True) + 1e-8)
        rollout = a @ rollout

    # CLS row, drop CLS column
    cls_rollout = rollout[0, 1:]  # (N,)
    grid = cls_rollout.reshape(GRID, GRID)
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    return grid


def attn_to_heatmap(grid: np.ndarray) -> np.ndarray:
    """Upsample (GRID, GRID) -> (224, 224) float32, values in [0,1]."""
    assert grid.shape == (GRID, GRID), f"Expected {GRID}x{GRID}, got {grid.shape}"
    t = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0)  # (1,1,14,14)
    up = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    out = up.squeeze().numpy()
    out = (out - out.min()) / (out.max() - out.min() + 1e-8)
    return out.astype(np.float32)


# ─── Localization metrics ─────────────────────────────────────────────────────

def _union_bbox(boxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    """Compute the union bounding box of a list of boxes."""
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return x1, y1, x2, y2


def pointing_game(heatmap: np.ndarray, boxes: List[Tuple[int,int,int,int]]) -> int:
    """1 if argmax pixel lies inside any GT box, else 0."""
    flat_idx = int(np.argmax(heatmap))
    py, px = divmod(flat_idx, IMG_SIZE)
    for x1, y1, x2, y2 in boxes:
        if x1 <= px <= x2 and y1 <= py <= y2:
            return 1
    return 0


def attention_mass_in_box(heatmap: np.ndarray, boxes: List[Tuple[int,int,int,int]]) -> float:
    """Fraction of total attention mass inside the union GT bbox."""
    total = heatmap.sum()
    if total < 1e-8:
        return 0.0
    x1, y1, x2, y2 = _union_bbox(boxes)
    inside = heatmap[y1:y2+1, x1:x2+1].sum()
    return float(inside / total)


def _threshold_bbox(heatmap: np.ndarray, threshold: float) -> Optional[Tuple[int,int,int,int]]:
    """Threshold heatmap, find largest connected foreground component, return its bbox."""
    binary = (heatmap >= threshold).astype(np.uint8)
    if binary.sum() == 0:
        return None

    labeled, n_components = scipy_label(binary)
    if n_components == 0:
        return None

    # Largest component
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background
    largest_label = int(sizes.argmax())
    component = (labeled == largest_label)

    ys, xs = np.where(component)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _iou(boxA: Tuple[int,int,int,int], boxB: Tuple[int,int,int,int]) -> float:
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1 + 1) * max(0, iy2 - iy1 + 1)
    areaA = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
    areaB = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    union = areaA + areaB - inter
    return float(inter / union) if union > 0 else 0.0


def bbox_iou_at_thresholds(
    heatmap: np.ndarray,
    boxes: List[Tuple[int,int,int,int]],
    thresholds: List[float] = THRESHOLDS,
) -> Dict[float, float]:
    """Compute IoU between predicted bbox (from thresholding) and union GT bbox."""
    gt_box = _union_bbox(boxes)
    result = {}
    for thr in thresholds:
        pred_box = _threshold_bbox(heatmap, thr)
        if pred_box is None:
            result[thr] = 0.0
        else:
            result[thr] = _iou(pred_box, gt_box)
    return result


# ─── Denormalization helper ────────────────────────────────────────────────────

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized CHW tensor back to HWC uint8 for visualization."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu().float() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


# ─── Visualization ────────────────────────────────────────────────────────────

def _draw_box(draw: ImageDraw.Draw, box: Tuple[int,int,int,int], color: str, width: int = 2):
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)


def save_sample_visualization(
    out_path: str,
    img_tensor: torch.Tensor,
    heatmap: np.ndarray,
    gt_boxes: List[Tuple[int,int,int,int]],
    pred_box: Optional[Tuple[int,int,int,int]],
    gt_class: int,
    pred_class: int,
    iou: float,
    pointing_hit: int,
    method: str,
    ckpt_name: str,
):
    img_np = denormalize(img_tensor)
    img_pil = Image.fromarray(img_np)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Left: original + GT boxes
    ax = axes[0]
    ax.imshow(img_np)
    ax.set_title(f"GT class: {gt_class} | Pred: {pred_class}", fontsize=8)
    for box in gt_boxes:
        rect = mpatches.Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            linewidth=2, edgecolor="lime", facecolor="none",
        )
        ax.add_patch(rect)
    if pred_box is not None:
        rect = mpatches.Rectangle(
            (pred_box[0], pred_box[1]), pred_box[2]-pred_box[0], pred_box[3]-pred_box[1],
            linewidth=2, edgecolor="red", facecolor="none", linestyle="--",
        )
        ax.add_patch(rect)
    ax.axis("off")

    # Middle: heatmap overlay
    ax = axes[1]
    ax.imshow(img_np)
    ax.imshow(heatmap, alpha=0.55, cmap="jet", vmin=0, vmax=1)
    ax.set_title(f"{method} | {ckpt_name}", fontsize=8)
    ax.axis("off")

    # Right: standalone heatmap
    ax = axes[2]
    im = ax.imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    ax.set_title(f"IoU={iou:.3f} | PG={'✓' if pointing_hit else '✗'}", fontsize=8)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ─── Streaming dataset for evaluation ─────────────────────────────────────────

class LocalizationValDataset:
    """Iterates over a fixed subset of ImageNet val images with annotations.

    Yields (img_tensor, label, val_index, original_pil_image) tuples.
    Skips images with no annotation.
    """

    def __init__(self, annotations: Dict[int, List], num_images: int, seed: int = 42):
        self.annotations = annotations
        self.num_images = num_images
        self.seed = seed

        # Determine which val indices to use
        available = sorted(annotations.keys())
        rng = np.random.RandomState(seed)
        if len(available) >= num_images:
            chosen = sorted(rng.choice(available, size=num_images, replace=False).tolist())
        else:
            print(
                f"[WARNING] Only {len(available)} annotated images available; "
                f"requested {num_images}. Using all."
            )
            chosen = available
        self.chosen_indices = chosen

    def __len__(self):
        return len(self.chosen_indices)

    def iterate(self):
        """Stream val set from HuggingFace, yielding items for chosen indices."""
        chosen_set = set(self.chosen_indices)
        max_idx = max(self.chosen_indices)

        ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)

        for i, ex in enumerate(ds):
            if i > max_idx:
                break
            if i not in chosen_set:
                continue

            img = ex["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")

            tensor = VAL_TRANSFORM(img)
            yield tensor, ex["label"], i, img


# ─── Per-checkpoint evaluation ────────────────────────────────────────────────

def evaluate_checkpoint(
    ckpt_name: str,
    ckpt_path: str,
    dataset: LocalizationValDataset,
    device: torch.device,
    output_dir: Path,
    attention_method: str,
    num_vis: int = 100,
    debug_vis: int = 20,
    log_every: int = 500,
) -> Dict:
    """Evaluate a single checkpoint on localization metrics.

    Returns a dict of aggregate metrics.
    """
    ckpt_out = output_dir / ckpt_name
    vis_dir = ckpt_out / "sample_visualizations"
    os.makedirs(ckpt_out, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    model = load_model(ckpt_path, device)
    all_block_indices = list(range(NUM_LAYERS))

    methods = []
    if attention_method in ("last", "both"):
        methods.append("last_layer")
    if attention_method in ("rollout", "both"):
        methods.append("rollout")

    per_image_rows = []      # rows for per_image_results.csv
    mass_rows = []           # rows for attention_mass_inside_box.csv
    vis_count = {m: 0 for m in methods}
    n_processed = 0

    # Running accumulators per method
    accum = {
        m: {
            "pointing": 0, "total": 0, "top1_correct": 0, "top5_correct": 0,
            "mass": [],
            "iou": {thr: [] for thr in THRESHOLDS},
            "iou_top1": {thr: [] for thr in THRESHOLDS},
        }
        for m in methods
    }

    print(f"\n{'='*60}")
    print(f"Evaluating: {ckpt_name}  ({ckpt_path})")
    print(f"{'='*60}")

    with torch.no_grad():
        for img_tensor, label, val_idx, pil_img in tqdm(
            dataset.iterate(), total=len(dataset), desc=ckpt_name, unit="img"
        ):
            gt_boxes = dataset.annotations[val_idx]
            if not gt_boxes:
                continue

            img_batch = img_tensor.unsqueeze(0).to(device)

            with capture_attention(model, all_block_indices) as get_attn:
                logits = model(img_batch)
                attn_dict = get_attn()

            # Classification
            probs = logits.squeeze(0)
            top5_preds = probs.topk(5).indices.cpu().tolist()
            pred_class = top5_preds[0]
            top1_correct = int(pred_class == label)
            top5_correct = int(label in top5_preds)

            # Extract grids per method
            grids: Dict[str, np.ndarray] = {}
            if "last_layer" in methods:
                grids["last_layer"] = _cls_attn_last_layer(attn_dict)
            if "rollout" in methods:
                grids["rollout"] = _attention_rollout(attn_dict)

            for m_name, grid in grids.items():
                heatmap = attn_to_heatmap(grid)
                pg      = pointing_game(heatmap, gt_boxes)
                mass    = attention_mass_in_box(heatmap, gt_boxes)
                ious    = bbox_iou_at_thresholds(heatmap, gt_boxes)
                best_iou = max(ious.values())
                pred_box = _threshold_bbox(heatmap, 0.5)  # for visualization

                acc = accum[m_name]
                acc["pointing"]    += pg
                acc["total"]       += 1
                acc["top1_correct"] += top1_correct
                acc["top5_correct"] += top5_correct
                acc["mass"].append(mass)
                for thr in THRESHOLDS:
                    acc["iou"][thr].append(ious[thr])
                    if top1_correct:
                        acc["iou_top1"][thr].append(ious[thr])

                per_image_rows.append({
                    "val_idx": val_idx,
                    "method": m_name,
                    "gt_class": label,
                    "pred_class": pred_class,
                    "top1_correct": top1_correct,
                    "top5_correct": top5_correct,
                    "pointing_game": pg,
                    "mass_in_box": round(mass, 6),
                    **{f"iou_thr{thr}": round(ious[thr], 6) for thr in THRESHOLDS},
                    "best_iou": round(best_iou, 6),
                    "corloc_05": int(best_iou >= 0.5),
                })
                mass_rows.append({
                    "val_idx": val_idx,
                    "method": m_name,
                    "mass_in_box": round(mass, 6),
                    "gt_class": label,
                })

                # Visualizations
                do_vis = (
                    vis_count[m_name] < num_vis
                    or (vis_count[m_name] < debug_vis and n_processed == 0)
                )
                if do_vis:
                    fname = f"{val_idx:08d}_{m_name}_iou{best_iou:.2f}_pg{pg}.png"
                    save_sample_visualization(
                        out_path=str(vis_dir / fname),
                        img_tensor=img_tensor,
                        heatmap=heatmap,
                        gt_boxes=gt_boxes,
                        pred_box=pred_box,
                        gt_class=label,
                        pred_class=pred_class,
                        iou=best_iou,
                        pointing_hit=pg,
                        method=m_name,
                        ckpt_name=ckpt_name,
                    )
                    vis_count[m_name] += 1

            n_processed += 1

            # Periodic logging
            if n_processed % log_every == 0:
                _print_running_metrics(ckpt_name, accum, methods, n_processed)

    # ── Final metrics ──
    summary = _compute_summary(ckpt_name, accum, methods)

    # Save CSVs
    _save_csv(per_image_rows, ckpt_out / "per_image_results.csv")
    _save_csv(mass_rows,      ckpt_out / "attention_mass_inside_box.csv")

    summary_rows = []
    for m_name, m_metrics in summary.items():
        row = {"method": m_name, **m_metrics}
        summary_rows.append(row)
    _save_csv(summary_rows, ckpt_out / "metrics_summary.csv")

    _print_running_metrics(ckpt_name, accum, methods, n_processed)
    return summary


def _print_running_metrics(
    ckpt_name: str,
    accum: dict,
    methods: List[str],
    n: int,
):
    for m_name in methods:
        a = accum[m_name]
        total = max(1, a["total"])
        pg_acc   = 100.0 * a["pointing"] / total
        top1_acc = 100.0 * a["top1_correct"] / total
        top5_acc = 100.0 * a["top5_correct"] / total
        mean_mass = 100.0 * float(np.mean(a["mass"])) if a["mass"] else 0.0
        mean_iou  = float(np.mean(a["iou"][0.5])) if a["iou"][0.5] else 0.0
        corloc    = 100.0 * float(np.mean([v >= 0.5 for v in a["iou"][0.5]])) if a["iou"][0.5] else 0.0
        print(
            f"[{ckpt_name}|{m_name}] n={n:,} | "
            f"Top1={top1_acc:.1f}% Top5={top5_acc:.1f}% | "
            f"PG={pg_acc:.1f}% | Mass={mean_mass:.1f}% | "
            f"mIoU@0.5={mean_iou:.3f} | CorLoc@0.5={corloc:.1f}%"
        )


def _compute_summary(ckpt_name: str, accum: dict, methods: List[str]) -> Dict:
    summary = {}
    for m_name in methods:
        a = accum[m_name]
        total = max(1, a["total"])
        pg_acc   = a["pointing"] / total
        top1_acc = a["top1_correct"] / total
        top5_acc = a["top5_correct"] / total
        mean_mass = float(np.mean(a["mass"])) if a["mass"] else 0.0

        iou_metrics = {}
        for thr in THRESHOLDS:
            vals = a["iou"][thr]
            iou_metrics[f"mean_iou_thr{thr}"] = float(np.mean(vals)) if vals else 0.0
            iou_metrics[f"corloc_thr{thr}"]   = float(np.mean([v >= thr for v in vals])) if vals else 0.0

        top1_iou_metrics = {}
        for thr in THRESHOLDS:
            vals = a["iou_top1"][thr]
            top1_iou_metrics[f"top1_mean_iou_thr{thr}"] = float(np.mean(vals)) if vals else 0.0
            top1_iou_metrics[f"top1_corloc_thr{thr}"]   = float(np.mean([v >= thr for v in vals])) if vals else 0.0

        summary[m_name] = {
            "checkpoint": ckpt_name,
            "n_images": total,
            "top1_accuracy": round(top1_acc, 6),
            "top5_accuracy": round(top5_acc, 6),
            "pointing_game_accuracy": round(pg_acc, 6),
            "mean_attention_mass_in_box": round(mean_mass, 6),
            **{k: round(v, 6) for k, v in iou_metrics.items()},
            **{k: round(v, 6) for k, v in top1_iou_metrics.items()},
        }
    return summary


def _save_csv(rows: List[dict], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Saved {path}")


# ─── Comparison plots ─────────────────────────────────────────────────────────

def make_comparison_plots(
    all_summaries: Dict[str, Dict],  # {ckpt_name: {method: metrics}}
    per_image_files: Dict[str, Path],  # {ckpt_name: per_image_results.csv path}
    output_dir: Path,
    methods: List[str],
):
    plots_dir = output_dir / "comparison_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    ckpt_names = list(all_summaries.keys())

    for method in methods:
        # Extract scalars per checkpoint
        pg_vals   = [all_summaries[c][method]["pointing_game_accuracy"] * 100  for c in ckpt_names if method in all_summaries[c]]
        mass_vals = [all_summaries[c][method]["mean_attention_mass_in_box"] * 100 for c in ckpt_names if method in all_summaries[c]]
        corloc_vals = [all_summaries[c][method]["corloc_thr0.5"] * 100 for c in ckpt_names if method in all_summaries[c]]
        ckpts_present = [c for c in ckpt_names if method in all_summaries[c]]

        def _bar(values, ylabel, title, fname):
            fig, ax = plt.subplots(figsize=(max(5, len(ckpts_present) * 1.5), 4))
            bars = ax.bar(ckpts_present, values, color=plt.cm.tab10.colors[:len(ckpts_present)])
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title} [{method}]")
            ax.set_ylim(0, max(values) * 1.25 if values else 1)
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=9,
                )
            plt.tight_layout()
            plt.savefig(plots_dir / fname, dpi=120)
            plt.close(fig)

        _bar(pg_vals,    "Pointing Game Accuracy (%)", "Pointing Game",          f"pointing_game_{method}.png")
        _bar(mass_vals,  "Attention Mass in GT Box (%)", "Attention Mass in Box", f"attn_mass_{method}.png")
        _bar(corloc_vals, "CorLoc @ IoU≥0.5 (%)", "CorLoc@0.5",                 f"corloc05_{method}.png")

    # ── Scatter: baseline IoU vs bimodal IoU per image ──
    _make_scatter_plots(per_image_files, plots_dir, methods)


def _load_per_image_csv(path: Path) -> Dict[Tuple[int, str], dict]:
    """Load per-image results CSV into a dict keyed by (val_idx, method)."""
    rows = {}
    if not path.exists():
        return rows
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (int(row["val_idx"]), row["method"])
            rows[key] = row
    return rows


def _make_scatter_plots(
    per_image_files: Dict[str, Path],
    plots_dir: Path,
    methods: List[str],
):
    ckpt_names = list(per_image_files.keys())
    if "baseline" not in ckpt_names:
        return

    baseline_data = _load_per_image_csv(per_image_files["baseline"])

    compare_against = [c for c in ckpt_names if c != "baseline"]
    thr = 0.5

    for method in methods:
        for other in compare_against:
            other_data = _load_per_image_csv(per_image_files[other])

            baseline_ious = []
            other_ious    = []
            common_keys = set(baseline_data) & set(other_data)
            common_keys = {k for k in common_keys if k[1] == method}

            for key in sorted(common_keys):
                b_iou = float(baseline_data[key][f"iou_thr{thr}"])
                o_iou = float(other_data[key][f"iou_thr{thr}"])
                baseline_ious.append(b_iou)
                other_ious.append(o_iou)

            if not baseline_ious:
                continue

            diffs = np.array(other_ious) - np.array(baseline_ious)

            # Scatter plot
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(baseline_ious, other_ious, alpha=0.3, s=8, c="steelblue")
            lim_max = max(max(baseline_ious), max(other_ious)) + 0.05
            ax.plot([0, lim_max], [0, lim_max], "r--", linewidth=1)
            ax.set_xlabel(f"Baseline IoU@{thr}")
            ax.set_ylabel(f"{other} IoU@{thr}")
            ax.set_title(f"Scatter: Baseline vs {other} [{method}]")
            plt.tight_layout()
            plt.savefig(plots_dir / f"scatter_baseline_vs_{other}_{method}.png", dpi=120)
            plt.close(fig)

            # Histogram of IoU differences
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(diffs, bins=40, color="steelblue", edgecolor="white")
            ax.axvline(0, color="red", linestyle="--", linewidth=1)
            ax.set_xlabel(f"IoU diff ({other} - baseline)")
            ax.set_ylabel("Count")
            ax.set_title(f"IoU diff histogram [{method}]")
            plt.tight_layout()
            plt.savefig(plots_dir / f"iou_diff_hist_{other}_{method}.png", dpi=120)
            plt.close(fig)

            print(f"[INFO] Saved scatter and histogram plots for {other} vs baseline [{method}]")


def _save_extreme_examples(
    per_image_files: Dict[str, Path],
    dataset: LocalizationValDataset,
    all_summaries: Dict,
    output_dir: Path,
    methods: List[str],
    n_examples: int = 10,
    device: torch.device = torch.device("cpu"),
):
    """Save extreme-improvement / extreme-degradation images (bimodal vs baseline)."""
    ckpt_names = list(per_image_files.keys())
    if "baseline" not in ckpt_names:
        return

    baseline_data = _load_per_image_csv(per_image_files["baseline"])
    compare_against = [c for c in ckpt_names if c != "baseline"]
    thr = 0.5

    for method in methods:
        for other in compare_against:
            other_data = _load_per_image_csv(per_image_files[other])
            common_keys = {k for k in set(baseline_data) & set(other_data) if k[1] == method}

            diffs = []
            for key in common_keys:
                b_iou = float(baseline_data[key][f"iou_thr{thr}"])
                o_iou = float(other_data[key][f"iou_thr{thr}"])
                diffs.append((o_iou - b_iou, key[0]))

            if not diffs:
                continue

            diffs.sort(key=lambda x: x[0])
            worst_n  = [idx for _, idx in diffs[:n_examples]]
            best_n   = [idx for _, idx in diffs[-n_examples:]]

            extreme_dir = output_dir / "comparison_plots" / f"extreme_{other}_vs_baseline_{method}"
            extreme_dir.mkdir(parents=True, exist_ok=True)

            target_indices = set(worst_n + best_n)
            buf: Dict[int, Tuple] = {}

            for img_tensor, label, val_idx, pil_img in dataset.iterate():
                if val_idx in target_indices:
                    buf[val_idx] = (img_tensor, label)
                if len(buf) == len(target_indices):
                    break

            for tag, indices in [("best_bimodal", best_n), ("worst_bimodal", worst_n)]:
                for idx in indices:
                    if idx not in buf:
                        continue
                    img_tensor, label = buf[idx]
                    diff_val = next(d for d, i in diffs if i == idx)
                    fname = f"{tag}_{idx:08d}_diff{diff_val:+.3f}.png"

                    b_row = baseline_data.get((idx, method), {})
                    o_row = other_data.get((idx, method), {})

                    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                    img_np = denormalize(img_tensor)
                    for ax, row, lbl in zip(axes, [b_row, o_row], ["baseline", other]):
                        ax.imshow(img_np)
                        ax.set_title(
                            f"{lbl}\nIoU@{thr}={float(row.get(f'iou_thr{thr}', 0)):.3f}",
                            fontsize=8,
                        )
                        ax.axis("off")
                    plt.suptitle(
                        f"GT={label} | Diff={diff_val:+.3f} | {method}",
                        fontsize=9,
                    )
                    plt.tight_layout()
                    plt.savefig(extreme_dir / fname, dpi=100)
                    plt.close(fig)

    print(f"[INFO] Saved extreme-example plots.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="ImageNet Localization evaluation for ViT-S checkpoints."
    )
    p.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        metavar="NAME:PATH",
        help=(
            "Checkpoint(s) to evaluate. Each entry must be 'name:path/to/best.pth'. "
            "E.g. --checkpoints baseline:runs/baseline_vit_s16_imagenet1k/checkpoints/best.pth"
        ),
    )
    p.add_argument(
        "--annotation-dir",
        default="data/imagenet_loc_annotations",
        help=(
            "Directory containing ILSVRC VOC-format XML annotation files. "
            "Expected layout: <dir>/ILSVRC2012_val_XXXXXXXX.xml "
            "(flat or one-level-deep in per-class subdirs)."
        ),
    )
    p.add_argument("--num-images",  type=int, default=2000,
                   help="Number of val images to evaluate (default: 2000). Use 50000 for full val.")
    p.add_argument("--output-dir",  default="runs/localization_eval",
                   help="Directory to save results and plots.")
    p.add_argument(
        "--attention-method",
        choices=["last", "rollout", "both"],
        default="both",
        help="Which attention extraction method(s) to use.",
    )
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-vis",     type=int, default=100,
                   help="Number of sample visualizations per checkpoint.")
    p.add_argument("--debug-vis",   type=int, default=20,
                   help="Number of debug visualizations (subset of num-vis).")
    p.add_argument("--log-every",   type=int, default=500,
                   help="Print running metrics every N images.")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # ── Parse checkpoints ──
    checkpoints: Dict[str, str] = {}
    for entry in args.checkpoints:
        if ":" not in entry:
            print(f"[ERROR] --checkpoints entries must be 'name:path', got: {entry!r}")
            sys.exit(1)
        name, path = entry.split(":", 1)
        if not os.path.exists(path):
            print(f"[ERROR] Checkpoint not found: {path!r}")
            sys.exit(1)
        checkpoints[name] = path

    print(f"[INFO] Checkpoints to evaluate: {list(checkpoints.keys())}")

    # ── Load annotations ──
    annotations = load_annotations(args.annotation_dir)

    # ── Build dataset (shared across all checkpoints) ──
    dataset = LocalizationValDataset(
        annotations=annotations,
        num_images=args.num_images,
        seed=args.seed,
    )
    print(f"[INFO] Evaluating on {len(dataset)} images.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    methods = []
    if args.attention_method in ("last", "both"):
        methods.append("last_layer")
    if args.attention_method in ("rollout", "both"):
        methods.append("rollout")

    all_summaries: Dict[str, Dict] = {}
    per_image_files: Dict[str, Path] = {}

    for ckpt_name, ckpt_path in checkpoints.items():
        summary = evaluate_checkpoint(
            ckpt_name=ckpt_name,
            ckpt_path=ckpt_path,
            dataset=dataset,
            device=device,
            output_dir=output_dir,
            attention_method=args.attention_method,
            num_vis=args.num_vis,
            debug_vis=args.debug_vis,
            log_every=args.log_every,
        )
        all_summaries[ckpt_name] = summary
        per_image_files[ckpt_name] = output_dir / ckpt_name / "per_image_results.csv"

    # ── Comparison plots ──
    make_comparison_plots(all_summaries, per_image_files, output_dir, methods)

    # ── Extreme examples ──
    _save_extreme_examples(
        per_image_files=per_image_files,
        dataset=dataset,
        all_summaries=all_summaries,
        output_dir=output_dir,
        methods=methods,
        n_examples=10,
        device=device,
    )

    # ── Save combined summary ──
    combined_rows = []
    for ckpt_name, method_summary in all_summaries.items():
        for method, metrics in method_summary.items():
            combined_rows.append({"checkpoint": ckpt_name, "method": method, **metrics})
    _save_csv(combined_rows, output_dir / "all_checkpoints_summary.csv")

    print("\n[DONE] Localization evaluation complete.")
    print(f"[INFO] Results saved to: {output_dir.resolve()}")

    # ── Print final table ──
    print("\n" + "="*80)
    print(f"{'Checkpoint':<18} {'Method':<12} {'Top1':>6} {'PG%':>8} {'Mass%':>8} {'CorLoc@0.5':>12}")
    print("="*80)
    for row in combined_rows:
        print(
            f"{row['checkpoint']:<18} {row['method']:<12} "
            f"{row['top1_accuracy']*100:>6.1f} "
            f"{row['pointing_game_accuracy']*100:>8.1f} "
            f"{row['mean_attention_mass_in_box']*100:>8.1f} "
            f"{row['corloc_thr0.5']*100:>12.1f}"
        )
    print("="*80)


if __name__ == "__main__":
    main()
