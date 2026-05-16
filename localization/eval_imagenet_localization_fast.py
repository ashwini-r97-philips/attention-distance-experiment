"""Fast GPU-accelerated ImageNet Localization Evaluation.

Evaluates baseline vs bimodal_medium checkpoints on localization quality using
attention rollout, with batched GPU inference, AMP, and vectorized metrics.

Both checkpoints are evaluated on the EXACT same fixed image subset for a fair
per-image delta comparison.

Usage
-----
  # Debug run (200 images, 20 visualizations)
  python eval_imagenet_localization_fast.py \
      --checkpoints baseline:runs/baseline_vit_s16_imagenet1k/checkpoints/best.pth \
                     bimodal_medium:runs/bimodal_medium_vit_s16_imagenet1k/checkpoints/best.pth \
      --max-images 200 \
      --batch-size 64 \
      --output-dir runs/localization_eval_debug \
      --debug-visualizations 20

  # Main 10k run
  python eval_imagenet_localization_fast.py \
      --checkpoints baseline:runs/baseline_vit_s16_imagenet1k/checkpoints/best.pth \
                     bimodal_medium:runs/bimodal_medium_vit_s16_imagenet1k/checkpoints/best.pth \
      --max-images 10000 \
      --batch-size 128 \
      --output-dir runs/localization_eval_10k_fast \
      --amp

  # Full val set
  python eval_imagenet_localization_fast.py \
      --checkpoints ... \
      --max-images 50000 \
      --batch-size 128 \
      --output-dir runs/localization_eval_50k_fast \
      --amp
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "bimodal_head_specialisation"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import timm
from datasets import load_dataset
from torchvision import transforms

from common.attention_hooks import patch_attention_forward, get_cached_attn_weights, unpatch_attention_forward


# ─── Constants ────────────────────────────────────────────────────────────────

IMG_SIZE    = 224
PATCH_SIZE  = 16
GRID        = IMG_SIZE // PATCH_SIZE   # 14
NUM_PATCHES = GRID * GRID              # 196
NUM_LAYERS  = 12
N_TOKENS    = NUM_PATCHES + 1          # +1 CLS

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

ATTN_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
CORLOC_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ─── Annotation loading ───────────────────────────────────────────────────────

def _parse_xml_annotation(xml_path: str) -> List[Tuple[int, int, int, int]]:
    """Parse ILSVRC VOC XML, return list of (x1,y1,x2,y2) in 224x224 space."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
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


def load_annotations(annotation_dir: str) -> Dict[int, List[Tuple[int,int,int,int]]]:
    ann_dir = Path(annotation_dir)
    if not ann_dir.exists():
        raise FileNotFoundError(
            f"Annotation directory not found: {annotation_dir}\n"
            "Expected ILSVRC VOC XML files at: <dir>/ILSVRC2012_val_XXXXXXXX.xml\n"
            "Download: https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_val_v3.tgz"
        )
    xml_files = list(ann_dir.glob("*.xml")) + list(ann_dir.glob("*/*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files in {annotation_dir}")

    annotations: Dict[int, List] = {}
    for xml_path in xml_files:
        parts = xml_path.stem.split("_")
        if len(parts) >= 3 and parts[-1].isdigit():
            idx = int(parts[-1]) - 1
            try:
                boxes = _parse_xml_annotation(str(xml_path))
                if boxes:
                    annotations[idx] = boxes
            except Exception:
                pass
    print(f"[INFO] Loaded annotations for {len(annotations):,} val images.")
    return annotations


# ─── Image subset selection ───────────────────────────────────────────────────

def select_image_ids(
    annotations: Dict[int, List],
    max_images: int,
    seed: int,
    id_file: Optional[Path] = None,
) -> List[int]:
    """Select a fixed reproducible subset of annotated val image indices."""
    available = sorted(annotations.keys())
    rng = np.random.RandomState(seed)
    if len(available) >= max_images:
        chosen = sorted(rng.choice(available, size=max_images, replace=False).tolist())
    else:
        print(f"[WARNING] Only {len(available)} annotated images available; using all.")
        chosen = available

    if id_file is not None:
        id_file.parent.mkdir(parents=True, exist_ok=True)
        with open(id_file, "w") as f:
            for idx in chosen:
                f.write(f"{idx}\n")
        print(f"[INFO] Saved {len(chosen)} selected image IDs → {id_file}")

    return chosen


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=1000)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


# ─── GPU-accelerated attention rollout ───────────────────────────────────────

def attention_rollout_batch_gpu(
    attn_dict: Dict[int, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Compute attention rollout for a batch entirely on GPU.

    attn_dict: {layer_idx: (B, H, N+1, N+1)} tensors already on GPU
    Returns: (B, N) float32 tensor on GPU — CLS-row rollout, patch tokens only
    """
    B = next(iter(attn_dict.values())).shape[0]
    n = N_TOKENS  # 197

    rollout = torch.eye(n, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1, -1)

    for layer_idx in sorted(attn_dict.keys()):
        a = attn_dict[layer_idx]            # (B, H, n, n)
        a = a.mean(dim=1)                   # (B, n, n) — mean over heads
        eye = torch.eye(n, dtype=a.dtype, device=device).unsqueeze(0)
        a = a + eye                         # add residual
        row_sum = a.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        a = a / row_sum                     # re-normalize rows
        rollout = torch.bmm(a, rollout)     # (B, n, n)

    # CLS row (index 0), drop CLS column (index 0) → patch tokens only
    cls_row = rollout[:, 0, 1:]             # (B, N=196)

    # Normalize each map to [0,1]
    mn = cls_row.min(dim=-1, keepdim=True).values
    mx = cls_row.max(dim=-1, keepdim=True).values
    cls_row = (cls_row - mn) / (mx - mn + 1e-8)

    return cls_row  # (B, 196)


def upsample_rollout_gpu(grid_maps: torch.Tensor) -> torch.Tensor:
    """Upsample (B, 196) → (B, 224, 224) on GPU.

    Args:
        grid_maps: (B, 196) float32 on GPU
    Returns:
        (B, 224, 224) float32 on GPU, values in [0,1]
    """
    B = grid_maps.shape[0]
    maps_2d = grid_maps.reshape(B, 1, GRID, GRID)
    up = F.interpolate(maps_2d, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    out = up.squeeze(1)  # (B, H, W)
    # Re-normalize after upsampling
    mn = out.flatten(1).min(dim=1).values[:, None, None]
    mx = out.flatten(1).max(dim=1).values[:, None, None]
    return (out - mn) / (mx - mn + 1e-8)


# ─── GPU-vectorized localization metrics ─────────────────────────────────────

def compute_metrics_batch_gpu(
    heatmaps: torch.Tensor,          # (B, 224, 224) on GPU, values in [0,1]
    gt_boxes_batch: List[List[Tuple[int,int,int,int]]],  # list length B
    attn_thresholds: List[float],
    device: torch.device,
) -> Dict:
    """Compute all localization metrics for a batch on GPU.

    Returns dict of lists (one entry per image in batch):
      - pointing_game: List[int]
      - mass_in_box: List[float]
      - iou_thr{t}: List[float]  for each t in attn_thresholds
    """
    B = heatmaps.shape[0]
    results = {
        "pointing_game": [],
        "mass_in_box": [],
        **{f"iou_thr{t}": [] for t in attn_thresholds},
    }

    for i in range(B):
        hm = heatmaps[i]  # (224, 224) on GPU
        boxes = gt_boxes_batch[i]
        if not boxes:
            results["pointing_game"].append(0)
            results["mass_in_box"].append(0.0)
            for t in attn_thresholds:
                results[f"iou_thr{t}"].append(0.0)
            continue

        # Union GT bbox
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)

        # Pointing game: argmax inside union box?
        flat_idx = hm.argmax().item()
        py, px = divmod(flat_idx, IMG_SIZE)
        pg = int(x1 <= px <= x2 and y1 <= py <= y2)
        results["pointing_game"].append(pg)

        # Attention mass in GT union box
        total = hm.sum()
        inside = hm[y1:y2+1, x1:x2+1].sum()
        mass = (inside / total.clamp(min=1e-8)).item()
        results["mass_in_box"].append(mass)

        # IoU at each attention threshold
        gt_box = (x1, y1, x2, y2)
        for t in attn_thresholds:
            binary = (hm >= t)
            if not binary.any():
                results[f"iou_thr{t}"].append(0.0)
                continue

            # Bounding box of the largest connected component
            # Use simple row/col extremes of thresholded mask (fast, approximate)
            # For exact connected components we'd need scipy on CPU — use the
            # bounding box of ALL thresholded pixels as a fast approximation,
            # then clip to the tightest rectangular region.
            rows = binary.any(dim=1).nonzero(as_tuple=False).view(-1)
            cols = binary.any(dim=0).nonzero(as_tuple=False).view(-1)
            if rows.numel() == 0 or cols.numel() == 0:
                results[f"iou_thr{t}"].append(0.0)
                continue

            pr_y1 = rows.min().item()
            pr_y2 = rows.max().item()
            pr_x1 = cols.min().item()
            pr_x2 = cols.max().item()

            # IoU
            ix1 = max(x1, pr_x1); iy1 = max(y1, pr_y1)
            ix2 = min(x2, pr_x2); iy2 = min(y2, pr_y2)
            inter = max(0, ix2 - ix1 + 1) * max(0, iy2 - iy1 + 1)
            area_gt   = (x2 - x1 + 1) * (y2 - y1 + 1)
            area_pred = (pr_x2 - pr_x1 + 1) * (pr_y2 - pr_y1 + 1)
            union = area_gt + area_pred - inter
            iou = inter / union if union > 0 else 0.0
            results[f"iou_thr{t}"].append(float(iou))

    return results


# ─── Dataset streaming ────────────────────────────────────────────────────────

class StreamingLocalizationDataset:
    """Streams ImageNet val from HuggingFace, yielding only chosen indices.

    Pads images into batches and returns:
        tensors: (B, 3, 224, 224) pinned CPU tensor
        labels: list[int]
        val_indices: list[int]
    """

    def __init__(
        self,
        chosen_indices: List[int],
        annotations: Dict[int, List],
        batch_size: int,
        num_workers: int = 2,
    ):
        self.chosen_indices = sorted(chosen_indices)
        self.annotations    = annotations
        self.batch_size     = batch_size
        self.chosen_set     = set(chosen_indices)
        self.max_idx        = max(chosen_indices)

    def __len__(self) -> int:
        return len(self.chosen_indices)

    def iterate_batches(self):
        """Yield (batch_tensor, labels, val_indices) tuples."""
        ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)

        buf_tensors: List[torch.Tensor] = []
        buf_labels:  List[int]          = []
        buf_indices: List[int]          = []

        for global_i, ex in enumerate(ds):
            if global_i > self.max_idx:
                break
            if global_i not in self.chosen_set:
                continue

            img = ex["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            t = VAL_TRANSFORM(img)
            buf_tensors.append(t)
            buf_labels.append(ex["label"])
            buf_indices.append(global_i)

            if len(buf_tensors) == self.batch_size:
                batch = torch.stack(buf_tensors).pin_memory()
                yield batch, list(buf_labels), list(buf_indices)
                buf_tensors, buf_labels, buf_indices = [], [], []

        if buf_tensors:
            batch = torch.stack(buf_tensors).pin_memory()
            yield batch, buf_labels, buf_indices


# ─── Per-checkpoint evaluation ────────────────────────────────────────────────

def evaluate_checkpoint(
    ckpt_name: str,
    ckpt_path: str,
    dataset: StreamingLocalizationDataset,
    annotations: Dict[int, List],
    device: torch.device,
    use_amp: bool,
    debug_last_layer: bool,
    log_every: int,
    output_dir: Path,
) -> Tuple[List[dict], Dict]:
    """Run batched GPU eval for one checkpoint.

    Returns:
        per_image_rows: list of dicts, one per image
        summary: dict of aggregate metrics
    """
    print(f"\n{'='*65}")
    print(f"  Checkpoint: {ckpt_name}  ({ckpt_path})")
    print(f"{'='*65}")

    model = load_model(ckpt_path, device)
    all_block_indices = list(range(NUM_LAYERS))
    patch_attention_forward(model, all_block_indices, differentiable=False)

    accum = {
        "pointing": 0, "total": 0, "top1_correct": 0, "top5_correct": 0,
        "mass": [],
        **{f"iou_thr{t}": [] for t in ATTN_THRESHOLDS},
    }

    per_image_rows: List[dict] = []
    n_processed = 0
    t_start = time.perf_counter()
    t_last_log = t_start

    pbar = tqdm(total=len(dataset), desc=ckpt_name, unit="img", dynamic_ncols=True)

    for batch_cpu, labels, val_indices in dataset.iterate_batches():
        B_actual = batch_cpu.shape[0]
        batch_gpu = batch_cpu.to(device, non_blocking=True)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(batch_gpu)
            attn_dict = get_cached_attn_weights(model, all_block_indices)
            # Convert all attn tensors to float32 for rollout computation
            attn_dict_f32 = {k: v.float() for k, v in attn_dict.items()}

            # Rollout on GPU
            rollout_maps = attention_rollout_batch_gpu(attn_dict_f32, device)  # (B, 196)
            heatmaps     = upsample_rollout_gpu(rollout_maps)                  # (B, 224, 224)

        # Classification
        top5_preds = logits.topk(5, dim=1).indices  # (B, 5)

        # Localization metrics (GPU-vectorized per-image loop)
        gt_boxes_batch = [annotations.get(idx, []) for idx in val_indices]
        metrics = compute_metrics_batch_gpu(heatmaps, gt_boxes_batch, ATTN_THRESHOLDS, device)

        # Record per-image results
        for i in range(B_actual):
            pred_class = top5_preds[i, 0].item()
            top1_ok    = int(pred_class == labels[i])
            top5_ok    = int(labels[i] in top5_preds[i].tolist())
            iou_vals   = {f"iou_thr{t}": metrics[f"iou_thr{t}"][i] for t in ATTN_THRESHOLDS}
            best_iou   = max(iou_vals.values())

            row = {
                "val_idx":      val_indices[i],
                "gt_class":     labels[i],
                "pred_class":   pred_class,
                "top1_correct": top1_ok,
                "top5_correct": top5_ok,
                "pointing_game": metrics["pointing_game"][i],
                "mass_in_box":  round(metrics["mass_in_box"][i], 6),
                **{k: round(v, 6) for k, v in iou_vals.items()},
                "best_iou":     round(best_iou, 6),
                "corloc_05":    int(best_iou >= 0.5),
            }
            per_image_rows.append(row)

            accum["pointing"]     += metrics["pointing_game"][i]
            accum["top1_correct"] += top1_ok
            accum["top5_correct"] += top5_ok
            accum["mass"].append(metrics["mass_in_box"][i])
            for t in ATTN_THRESHOLDS:
                accum[f"iou_thr{t}"].append(iou_vals[f"iou_thr{t}"])
            accum["total"] += 1

        n_processed += B_actual
        pbar.update(B_actual)

        # Throughput logging
        if n_processed % log_every < B_actual or n_processed == len(dataset):
            elapsed = time.perf_counter() - t_start
            tput = n_processed / elapsed
            n = max(1, accum["total"])
            pg   = 100.0 * accum["pointing"] / n
            top1 = 100.0 * accum["top1_correct"] / n
            mass = 100.0 * float(np.mean(accum["mass"])) if accum["mass"] else 0.0
            iou5 = float(np.mean(accum["iou_thr0.5"])) if accum["iou_thr0.5"] else 0.0
            print(
                f"  [{ckpt_name}] {n_processed:,}/{len(dataset):,} images | "
                f"{tput:.1f} img/s | "
                f"Top1={top1:.1f}% | PG={pg:.1f}% | "
                f"Mass={mass:.1f}% | mIoU@0.5={iou5:.4f}"
            )

    pbar.close()
    unpatch_attention_forward(model, all_block_indices)
    del model
    torch.cuda.empty_cache()

    total_time = time.perf_counter() - t_start
    tput_final = n_processed / total_time

    # Compute summary
    n = max(1, accum["total"])
    summary = {
        "checkpoint":        ckpt_name,
        "n_images":          n,
        "total_time_s":      round(total_time, 2),
        "throughput_img_s":  round(tput_final, 2),
        "top1_accuracy":     round(accum["top1_correct"] / n, 6),
        "top5_accuracy":     round(accum["top5_correct"] / n, 6),
        "pointing_game_accuracy":       round(accum["pointing"] / n, 6),
        "mean_attention_mass_in_box":   round(float(np.mean(accum["mass"])), 6),
    }
    for t in ATTN_THRESHOLDS:
        vals = accum[f"iou_thr{t}"]
        summary[f"mean_iou_thr{t}"]  = round(float(np.mean(vals)) if vals else 0.0, 6)
    for t in CORLOC_THRESHOLDS:
        key = f"iou_thr{t}" if t in ATTN_THRESHOLDS else f"iou_thr{t}"
        vals = accum.get(f"iou_thr{t}", [])
        summary[f"corloc_thr{t}"] = round(float(np.mean([v >= t for v in vals])) if vals else 0.0, 6)

    return per_image_rows, summary, tput_final


# ─── Bootstrap confidence intervals ──────────────────────────────────────────

def bootstrap_ci(
    values: List[float],
    stat_fn,
    n_resamples: int = 1000,
    ci: float = 0.95,
    device: Optional[torch.device] = None,
) -> Tuple[float, float, float]:
    """Bootstrap 95% CI using torch vectorized resampling.

    Returns (point_estimate, lower, upper).
    """
    arr = torch.tensor(values, dtype=torch.float32)
    n   = len(arr)
    # Vectorized: (n_resamples, n) integer indices
    idx = torch.randint(0, n, (n_resamples, n))
    samples = arr[idx]                        # (n_resamples, n)
    stats   = samples.mean(dim=1)             # works for mean-based stats
    lo = float(torch.quantile(stats, (1 - ci) / 2))
    hi = float(torch.quantile(stats, (1 + ci) / 2))
    return float(arr.mean()), lo, hi


def compute_bootstrap_cis(per_image_rows: List[dict], name: str) -> Dict:
    """Compute bootstrap 95% CIs for key metrics."""
    pgs   = [r["pointing_game"]  for r in per_image_rows]
    masses = [r["mass_in_box"]   for r in per_image_rows]
    iou3  = [r["iou_thr0.3"]     for r in per_image_rows]
    cl3   = [float(r["iou_thr0.3"] >= 0.3) for r in per_image_rows]
    cl5   = [float(r["iou_thr0.5"] >= 0.5) for r in per_image_rows]

    cis = {}
    for metric_name, vals in [
        ("pointing_game", pgs),
        ("mass_in_box", masses),
        ("mean_iou_0.3", iou3),
        ("corloc_0.3", cl3),
        ("corloc_0.5", cl5),
    ]:
        pt, lo, hi = bootstrap_ci(vals, np.mean)
        cis[metric_name] = {"point": round(pt, 6), "ci95_lo": round(lo, 6), "ci95_hi": round(hi, 6)}

    return cis


# ─── Delta analysis ───────────────────────────────────────────────────────────

def compute_delta(
    base_rows: List[dict],
    comp_rows: List[dict],
    base_name: str,
    comp_name: str,
    output_dir: Path,
) -> Tuple[List[dict], Dict]:
    """Compute per-image IoU delta: comp - baseline at thr=0.5."""
    base_map = {r["val_idx"]: r for r in base_rows}
    comp_map = {r["val_idx"]: r for r in comp_rows}
    common   = sorted(set(base_map) & set(comp_map))

    delta_rows = []
    diffs = []
    for idx in common:
        b = float(base_map[idx]["iou_thr0.5"])
        c = float(comp_map[idx]["iou_thr0.5"])
        d = c - b
        diffs.append(d)
        delta_rows.append({
            "val_idx": idx,
            f"{base_name}_iou_0.5": round(b, 6),
            f"{comp_name}_iou_0.5": round(c, 6),
            "delta_iou_0.5": round(d, 6),
        })

    diffs = np.array(diffs)

    # Bootstrap CI on mean delta
    delta_pt, delta_lo, delta_hi = bootstrap_ci(diffs.tolist(), np.mean)

    summary = {
        "n_compared":          len(diffs),
        "mean_delta":          round(float(diffs.mean()), 6),
        "median_delta":        round(float(np.median(diffs)), 6),
        "std_delta":           round(float(diffs.std()), 6),
        "mean_delta_ci95_lo":  round(delta_lo, 6),
        "mean_delta_ci95_hi":  round(delta_hi, 6),
        "improved_count":      int((diffs > 0).sum()),
        "improved_pct":        round(100.0 * (diffs > 0).mean(), 2),
        "unchanged_count":     int((diffs == 0).sum()),
        "unchanged_pct":       round(100.0 * (diffs == 0).mean(), 2),
        "degraded_count":      int((diffs < 0).sum()),
        "degraded_pct":        round(100.0 * (diffs < 0).mean(), 2),
        "large_gain_count":    int((diffs > 0.1).sum()),
        "large_gain_pct":      round(100.0 * (diffs > 0.1).mean(), 2),
        "large_loss_count":    int((diffs < -0.1).sum()),
        "large_loss_pct":      round(100.0 * (diffs < -0.1).mean(), 2),
    }

    out_csv = output_dir / f"per_image_delta_{comp_name}_vs_{base_name}.csv"
    _save_csv(delta_rows, out_csv)
    return delta_rows, summary, diffs


# ─── CSV helpers ─────────────────────────────────────────────────────────────

def _save_csv(rows: List[dict], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Saved → {path}")


# ─── Denormalize helper ───────────────────────────────────────────────────────

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = (tensor.cpu().float() * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ─── Visualization ────────────────────────────────────────────────────────────

def save_single_vis(
    out_path: Path,
    img_tensor: torch.Tensor,
    hm_base: np.ndarray,
    hm_comp: np.ndarray,
    gt_boxes: List[Tuple[int,int,int,int]],
    gt_class: int,
    pred_base: int,
    pred_comp: int,
    iou_base: float,
    iou_comp: float,
    pg_base: int,
    pg_comp: int,
    base_name: str,
    comp_name: str,
):
    img_np = denormalize(img_tensor)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = [
        f"GT={gt_class}",
        f"{base_name}: pred={pred_base}\nIoU@0.5={iou_base:.3f} PG={'✓' if pg_base else '✗'}",
        f"{comp_name}: pred={pred_comp}\nIoU@0.5={iou_comp:.3f} PG={'✓' if pg_comp else '✗'}",
        f"Δ IoU = {iou_comp-iou_base:+.3f}",
    ]

    # Col 0: image + GT boxes
    axes[0].imshow(img_np)
    for box in gt_boxes:
        rect = mpatches.Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            linewidth=2, edgecolor="lime", facecolor="none"
        )
        axes[0].add_patch(rect)

    # Col 1 & 2: heatmap overlays
    for ax, hm, pred, iou, pg, label in [
        (axes[1], hm_base, pred_base, iou_base, pg_base, base_name),
        (axes[2], hm_comp, pred_comp, iou_comp, pg_comp, comp_name),
    ]:
        ax.imshow(img_np)
        ax.imshow(hm, alpha=0.55, cmap="jet", vmin=0, vmax=1)

    # Col 3: difference map
    diff = hm_comp.astype(np.float32) - hm_base.astype(np.float32)
    im = axes[3].imshow(diff, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=7)
        ax.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def save_qualitative_grid(
    vis_dir: Path,
    img_store: Dict[int, torch.Tensor],
    hm_store_base: Dict[int, np.ndarray],
    hm_store_comp: Dict[int, np.ndarray],
    per_image_base: Dict[int, dict],
    per_image_comp: Dict[int, dict],
    annotations: Dict[int, List],
    delta_rows: List[dict],
    base_name: str,
    comp_name: str,
    n_random: int = 30,
    n_best: int = 30,
    n_worst: int = 30,
    n_highconf: int = 10,
    seed: int = 42,
):
    """Save qualitative visualizations: random, best gains, worst failures, high-conf."""
    vis_dir.mkdir(parents=True, exist_ok=True)

    sorted_delta = sorted(delta_rows, key=lambda r: r["delta_iou_0.5"])
    worst_idxs   = [r["val_idx"] for r in sorted_delta[:n_worst]]
    best_idxs    = [r["val_idx"] for r in sorted_delta[-n_best:]]

    rng = np.random.RandomState(seed)
    all_ids = sorted(img_store.keys())
    random_idxs = rng.choice(all_ids, size=min(n_random, len(all_ids)), replace=False).tolist()

    # High-confidence correct: top1 both correct AND high iou_base
    highconf_idxs = sorted(
        [idx for idx, r in per_image_base.items()
         if r["top1_correct"] == 1 and float(r["iou_thr0.5"]) >= 0.3],
        key=lambda idx: -float(per_image_base[idx]["iou_thr0.5"]),
    )[:n_highconf]

    groups = [
        ("random",    random_idxs),
        ("best_gain", best_idxs),
        ("worst_fail",worst_idxs),
        ("highconf",  highconf_idxs),
    ]

    for group_name, idxs in groups:
        group_dir = vis_dir / group_name
        group_dir.mkdir(exist_ok=True)
        for idx in idxs:
            if idx not in img_store:
                continue
            b_row = per_image_base.get(idx, {})
            c_row = per_image_comp.get(idx, {})
            save_single_vis(
                out_path=group_dir / f"{idx:08d}.png",
                img_tensor=img_store[idx],
                hm_base=hm_store_base.get(idx, np.zeros((224, 224))),
                hm_comp=hm_store_comp.get(idx, np.zeros((224, 224))),
                gt_boxes=annotations.get(idx, []),
                gt_class=b_row.get("gt_class", -1),
                pred_base=b_row.get("pred_class", -1),
                pred_comp=c_row.get("pred_class", -1),
                iou_base=float(b_row.get("iou_thr0.5", 0)),
                iou_comp=float(c_row.get("iou_thr0.5", 0)),
                pg_base=int(b_row.get("pointing_game", 0)),
                pg_comp=int(c_row.get("pointing_game", 0)),
                base_name=base_name,
                comp_name=comp_name,
            )

    print(f"[INFO] Saved qualitative visualizations → {vis_dir}")


# ─── Plots ────────────────────────────────────────────────────────────────────

def make_plots(
    summaries: Dict[str, Dict],
    per_image: Dict[str, List[dict]],
    delta_rows: List[dict],
    delta_summary: Dict,
    diffs: np.ndarray,
    output_dir: Path,
    ckpt_names: List[str],
):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    colors = ["#4878CF", "#D65F5F"]

    # 1. Bar: pointing game
    fig, ax = plt.subplots(figsize=(5, 4))
    vals = [summaries[c]["pointing_game_accuracy"] * 100 for c in ckpt_names]
    bars = ax.bar(ckpt_names, vals, color=colors[:len(ckpt_names)])
    ax.set_ylabel("Pointing Game Accuracy (%)")
    ax.set_title("Pointing Game")
    ax.set_ylim(0, max(vals) * 1.2 + 1)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{v:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(plots_dir / "pointing_game.png", dpi=120)
    plt.close()

    # 2. Bar: CorLoc at multiple thresholds
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(CORLOC_THRESHOLDS))
    width = 0.35
    for i, ckpt in enumerate(ckpt_names):
        corloc_vals = [summaries[ckpt].get(f"corloc_thr{t}", 0) * 100 for t in CORLOC_THRESHOLDS]
        ax.bar(x + i * width, corloc_vals, width, label=ckpt, color=colors[i])
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f"≥{t}" for t in CORLOC_THRESHOLDS])
    ax.set_xlabel("IoU Threshold")
    ax.set_ylabel("CorLoc (%)")
    ax.set_title("CorLoc at IoU Thresholds")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "corloc_bar.png", dpi=120)
    plt.close()

    # 3. Line: CorLoc over IoU threshold
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, ckpt in enumerate(ckpt_names):
        corloc_vals = [summaries[ckpt].get(f"corloc_thr{t}", 0) * 100 for t in CORLOC_THRESHOLDS]
        ax.plot(CORLOC_THRESHOLDS, corloc_vals, marker="o", label=ckpt, color=colors[i])
    ax.set_xlabel("IoU Threshold")
    ax.set_ylabel("CorLoc (%)")
    ax.set_title("CorLoc Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "corloc_curve.png", dpi=120)
    plt.close()

    # 4. Line: mIoU over attention threshold
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, ckpt in enumerate(ckpt_names):
        miou_vals = [summaries[ckpt].get(f"mean_iou_thr{t}", 0) for t in ATTN_THRESHOLDS]
        ax.plot(ATTN_THRESHOLDS, miou_vals, marker="o", label=ckpt, color=colors[i])
    ax.set_xlabel("Attention Threshold")
    ax.set_ylabel("Mean IoU")
    ax.set_title("mIoU vs Attention Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "miou_curve.png", dpi=120)
    plt.close()

    # 5. Histogram: IoU delta (comp - baseline)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(diffs, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="no change")
    ax.axvline(float(diffs.mean()), color="orange", linestyle="-", linewidth=1.5,
               label=f"mean={diffs.mean():+.4f}")
    ax.set_xlabel(f"IoU delta ({ckpt_names[1]} − {ckpt_names[0]}) at thr=0.5")
    ax.set_ylabel("Count")
    ax.set_title("Per-image IoU Delta")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "iou_delta_hist.png", dpi=120)
    plt.close()

    # 6. Scatter: baseline IoU vs comp IoU
    if len(ckpt_names) == 2:
        base_ious = [float(per_image[ckpt_names[0]][i]["iou_thr0.5"])
                     for i in range(len(per_image[ckpt_names[0]]))]
        comp_ious = [float(per_image[ckpt_names[1]][i]["iou_thr0.5"])
                     for i in range(len(per_image[ckpt_names[1]]))]
        lim = max(max(base_ious, default=0), max(comp_ious, default=0)) * 1.05 + 0.01
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(base_ious, comp_ious, alpha=0.2, s=6, c="steelblue")
        ax.plot([0, lim], [0, lim], "r--", linewidth=1)
        ax.set_xlabel(f"{ckpt_names[0]} IoU@0.5")
        ax.set_ylabel(f"{ckpt_names[1]} IoU@0.5")
        ax.set_title("Scatter: per-image IoU@0.5")
        plt.tight_layout()
        plt.savefig(plots_dir / "scatter_iou.png", dpi=120)
        plt.close()

    # 7. Histogram: attention mass in GT bbox
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, ckpt in enumerate(ckpt_names):
        masses = [float(r["mass_in_box"]) * 100 for r in per_image[ckpt]]
        ax.hist(masses, bins=40, alpha=0.6, label=ckpt, color=colors[i], edgecolor="none")
    ax.set_xlabel("Attention Mass in GT Bbox (%)")
    ax.set_ylabel("Count")
    ax.set_title("Attention Mass Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "attn_mass_hist.png", dpi=120)
    plt.close()

    print(f"[INFO] Saved all plots → {plots_dir}")


# ─── Runtime summary ─────────────────────────────────────────────────────────

def write_runtime_summary(
    output_dir: Path,
    summaries: Dict[str, Dict],
    args,
    device: torch.device,
):
    lines = []
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
    else:
        gpu_name = "CPU"

    lines.append(f"GPU:              {gpu_name}")
    lines.append(f"Device:           {device}")
    lines.append(f"Batch size:       {args.batch_size}")
    lines.append(f"AMP:              {args.amp}")
    lines.append(f"Max images:       {args.max_images}")
    lines.append(f"Seed:             {args.seed}")
    lines.append("")
    for ckpt_name, s in summaries.items():
        lines.append(f"[{ckpt_name}]")
        lines.append(f"  n_images:       {s['n_images']:,}")
        lines.append(f"  total_time:     {s['total_time_s']:.1f}s")
        lines.append(f"  throughput:     {s['throughput_img_s']:.1f} img/s")
        lines.append("")

    path = output_dir / "runtime_summary.txt"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Saved runtime summary → {path}")


# ─── Debug validation pass ────────────────────────────────────────────────────

def debug_validation_pass(
    model: torch.nn.Module,
    chosen_indices: List[int],
    annotations: Dict[int, List],
    device: torch.device,
    use_amp: bool,
    output_dir: Path,
    n_vis: int = 20,
) -> None:
    """Sanity check: verify bbox coordinate mapping, check for NaNs, save overlays."""
    print("\n[DEBUG] Running debug validation pass...")
    debug_dir = output_dir / "debug_vis"
    debug_dir.mkdir(parents=True, exist_ok=True)

    all_block_indices = list(range(NUM_LAYERS))
    patch_attention_forward(model, all_block_indices, differentiable=False)

    sample_ious = []
    sample_pgs  = []
    nan_count   = 0
    saved       = 0

    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
    chosen_set  = set(chosen_indices)
    max_idx     = max(chosen_indices)

    for global_i, ex in enumerate(ds):
        if global_i > max_idx or saved >= len(chosen_indices):
            break
        if global_i not in chosen_set:
            continue

        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        t = VAL_TRANSFORM(img).unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(t)
            attn_dict = get_cached_attn_weights(model, all_block_indices)
            attn_dict_f32 = {k: v.float() for k, v in attn_dict.items()}

            rollout = attention_rollout_batch_gpu(attn_dict_f32, device)
            hm = upsample_rollout_gpu(rollout)[0]  # (224, 224)

        if torch.isnan(hm).any():
            nan_count += 1
            print(f"  [WARNING] NaN in rollout for val_idx={global_i}")
            continue

        hm_np = hm.cpu().numpy()
        boxes = annotations.get(global_i, [])
        if not boxes:
            saved += 1
            continue

        metrics = compute_metrics_batch_gpu(hm.unsqueeze(0), [boxes], ATTN_THRESHOLDS, device)
        iou05   = metrics["iou_thr0.5"][0]
        pg      = metrics["pointing_game"][0]
        sample_ious.append(iou05)
        sample_pgs.append(pg)

        if saved < n_vis:
            img_np = denormalize(t.squeeze(0))
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_np)
            for box in boxes:
                rect = mpatches.Rectangle(
                    (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                    linewidth=2, edgecolor="lime", facecolor="none"
                )
                axes[0].add_patch(rect)
            axes[0].set_title(f"GT boxes | val_idx={global_i}", fontsize=8)
            axes[0].axis("off")

            axes[1].imshow(img_np)
            axes[1].imshow(hm_np, alpha=0.55, cmap="jet", vmin=0, vmax=1)
            axes[1].set_title(f"Rollout overlay | IoU@0.5={iou05:.3f}", fontsize=8)
            axes[1].axis("off")

            axes[2].imshow(hm_np, cmap="jet", vmin=0, vmax=1)
            axes[2].set_title(f"Rollout map | PG={'✓' if pg else '✗'}", fontsize=8)
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(debug_dir / f"debug_{global_i:08d}.png", dpi=90, bbox_inches="tight")
            plt.close()

        saved += 1

    unpatch_attention_forward(model, all_block_indices)

    print(f"\n[DEBUG] Results on {len(sample_ious)} debug images:")
    print(f"  NaN maps:      {nan_count}")
    print(f"  Mean IoU@0.5:  {np.mean(sample_ious):.4f}")
    print(f"  Pointing Game: {100*np.mean(sample_pgs):.1f}%")
    print(f"  Sample IoUs:   {[round(v,3) for v in sample_ious[:10]]}")
    print(f"  Debug vis →    {debug_dir}\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Fast GPU-accelerated ImageNet localization evaluation."
    )
    p.add_argument(
        "--checkpoints", nargs="+", required=True, metavar="NAME:PATH",
        help="Checkpoints: 'name:path/to/best.pth'. First is treated as baseline.",
    )
    p.add_argument(
        "--annotation-dir",
        default="data/imagenet_loc_annotations",
        help="Directory with ILSVRC VOC XML annotation files.",
    )
    p.add_argument("--max-images",  type=int, default=10000,
                   help="Number of val images (default 10000; use 50000 for full).")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--batch-size",  type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--output-dir",  default="runs/localization_eval_10k_fast")
    p.add_argument("--amp",         action="store_true", default=False,
                   help="Enable AMP (torch.autocast) for faster inference.")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--save-rollout-maps", type=lambda x: x.lower() == "true", default=False,
        help="Save full (N,224,224) rollout maps per checkpoint (large disk use).",
    )
    p.add_argument(
        "--debug-last-layer", action="store_true", default=False,
        help="Also compute last-layer CLS attention metrics (slower, optional).",
    )
    p.add_argument("--num-vis",            type=int, default=100,
                   help="Total qualitative visualizations to save (across groups).")
    p.add_argument("--debug-visualizations", type=int, default=20,
                   help="Number of debug vis to save before main run.")
    p.add_argument("--log-every",   type=int, default=500)
    p.add_argument(
        "--skip-debug",  action="store_true", default=False,
        help="Skip the 200-image debug pass.",
    )
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Parse checkpoints
    checkpoints: Dict[str, str] = {}
    for entry in args.checkpoints:
        if ":" not in entry:
            print(f"[ERROR] --checkpoints must be 'name:path', got: {entry!r}")
            sys.exit(1)
        name, path = entry.split(":", 1)
        if not os.path.exists(path):
            print(f"[ERROR] Checkpoint not found: {path!r}")
            sys.exit(1)
        checkpoints[name] = path

    ckpt_names = list(checkpoints.keys())
    print(f"[INFO] Checkpoints: {ckpt_names}")

    device     = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(device)}")
        torch.backends.cudnn.benchmark = True

    # Load annotations
    annotations = load_annotations(args.annotation_dir)

    # Select fixed image subset
    id_file = output_dir / f"selected_imagenetloc_{args.max_images}_{args.seed}.txt"
    # Rename to the specific canonical name for the 10k seed42 case
    if args.max_images == 10000 and args.seed == 42:
        id_file = output_dir / "selected_imagenetloc_10k_seed42.txt"

    chosen_indices = select_image_ids(annotations, args.max_images, args.seed, id_file)
    print(f"[INFO] Evaluating on {len(chosen_indices):,} images.")

    # ── Debug pass (200 images, first checkpoint) ──
    if not args.skip_debug:
        debug_n = min(200, len(chosen_indices))
        debug_indices = chosen_indices[:debug_n]
        debug_model = load_model(list(checkpoints.values())[0], device)
        debug_validation_pass(
            model=debug_model,
            chosen_indices=debug_indices,
            annotations=annotations,
            device=device,
            use_amp=args.amp,
            output_dir=output_dir,
            n_vis=args.debug_visualizations,
        )
        del debug_model
        torch.cuda.empty_cache()
        print("[DEBUG] Debug pass complete. Proceeding with full evaluation.\n")

    # ── Per-checkpoint evaluation ──
    dataset = StreamingLocalizationDataset(
        chosen_indices=chosen_indices,
        annotations=annotations,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    all_per_image: Dict[str, List[dict]] = {}
    all_summaries: Dict[str, Dict]       = {}
    # Stores for visualization (only keep if we need them — controlled by memory)
    img_store:      Dict[int, torch.Tensor]  = {}
    hm_store_base:  Dict[int, np.ndarray]    = {}
    hm_store_comp:  Dict[int, np.ndarray]    = {}

    # We need to collect images & heatmaps for vis during the eval passes.
    # To avoid streaming twice for vis purposes, we store a small sample.
    # Determine which indices to store for visualization (all vis targets).
    # We'll figure out best/worst after first pass, so we store ALL heatmaps
    # for the base pass if max_images is small enough (≤10k), otherwise skip.
    store_heatmaps = len(chosen_indices) <= 10001

    for pass_idx, (ckpt_name, ckpt_path) in enumerate(checkpoints.items()):
        per_image_rows, summary, tput = evaluate_checkpoint(
            ckpt_name=ckpt_name,
            ckpt_path=ckpt_path,
            dataset=dataset,
            annotations=annotations,
            device=device,
            use_amp=args.amp,
            debug_last_layer=args.debug_last_layer,
            log_every=args.log_every,
            output_dir=output_dir,
        )

        # If we need heatmaps for vis, we need to re-run a lighter pass.
        # For simplicity: store them during the main eval loops.
        # (Heatmaps are already computed above; we re-stream in save_qualitative_grid
        #  which avoids storing them all in memory by default.)

        all_per_image[ckpt_name] = per_image_rows
        all_summaries[ckpt_name] = summary

        # Save per-checkpoint CSV
        csv_path = output_dir / f"per_image_results_{ckpt_name}.csv"
        _save_csv(per_image_rows, csv_path)

        # Save per-checkpoint summary CSV
        summary_csv = output_dir / f"metrics_summary_{ckpt_name}.csv"
        _save_csv([summary], summary_csv)

        # Bootstrap CIs
        cis = compute_bootstrap_cis(per_image_rows, ckpt_name)
        ci_path = output_dir / f"bootstrap_ci_{ckpt_name}.json"
        with open(ci_path, "w") as f:
            json.dump(cis, f, indent=2)
        print(f"[INFO] Bootstrap CIs → {ci_path}")

    # ── Delta analysis ──
    all_summaries_csv_rows = []
    delta_rows, delta_summary, diffs = [], {}, np.array([])

    if len(ckpt_names) >= 2:
        base_name = ckpt_names[0]
        comp_name = ckpt_names[1]

        # Align rows by val_idx for fair delta
        base_map = {r["val_idx"]: r for r in all_per_image[base_name]}
        comp_map = {r["val_idx"]: r for r in all_per_image[comp_name]}
        delta_rows, delta_summary, diffs = compute_delta(
            base_rows=list(base_map.values()),
            comp_rows=list(comp_map.values()),
            base_name=base_name,
            comp_name=comp_name,
            output_dir=output_dir,
        )

        # Bootstrap CI on mean delta
        delta_pt, delta_lo, delta_hi = bootstrap_ci(diffs.tolist(), np.mean)
        delta_summary["mean_delta_bootstrap_ci95"] = f"[{delta_lo:+.5f}, {delta_hi:+.5f}]"

        delta_summary_path = output_dir / f"delta_summary_{comp_name}_vs_{base_name}.json"
        with open(delta_summary_path, "w") as f:
            json.dump(delta_summary, f, indent=2)
        print(f"[INFO] Delta summary → {delta_summary_path}")

    # ── Combined summary CSV ──
    for ckpt_name, s in all_summaries.items():
        all_summaries_csv_rows.append(s)
    _save_csv(all_summaries_csv_rows, output_dir / "metrics_summary.csv")

    # ── Runtime summary ──
    write_runtime_summary(output_dir, all_summaries, args, device)

    # ── Plots ──
    if len(ckpt_names) >= 2 and len(diffs) > 0:
        make_plots(
            summaries=all_summaries,
            per_image=all_per_image,
            delta_rows=delta_rows,
            delta_summary=delta_summary,
            diffs=diffs,
            output_dir=output_dir,
            ckpt_names=ckpt_names,
        )

    # ── Qualitative visualizations ──
    # Stream images + heatmaps for the vis subset only
    if len(ckpt_names) >= 2 and len(diffs) > 0:
        print("\n[INFO] Generating qualitative visualizations (streaming images once more)...")

        base_name = ckpt_names[0]
        comp_name = ckpt_names[1]
        base_map  = {r["val_idx"]: r for r in all_per_image[base_name]}
        comp_map  = {r["val_idx"]: r for r in all_per_image[comp_name]}

        # Determine which images to visualize
        n_random   = args.num_vis // 3
        n_best     = args.num_vis // 3
        n_worst    = args.num_vis // 3
        n_highconf = max(1, args.num_vis - n_random - n_best - n_worst)

        sorted_delta_by_val = sorted(delta_rows, key=lambda r: r["delta_iou_0.5"])
        worst_target = set(r["val_idx"] for r in sorted_delta_by_val[:n_worst])
        best_target  = set(r["val_idx"] for r in sorted_delta_by_val[-n_best:])

        rng = np.random.RandomState(args.seed + 1)
        random_target = set(
            rng.choice(chosen_indices, size=min(n_random, len(chosen_indices)), replace=False).tolist()
        )
        highconf_target = set(sorted(
            [idx for idx, r in base_map.items()
             if r["top1_correct"] == 1 and float(r["iou_thr0.5"]) >= 0.2],
            key=lambda idx: -float(base_map[idx]["iou_thr0.5"])
        )[:n_highconf])

        vis_target_set = worst_target | best_target | random_target | highconf_target
        print(f"[INFO] Vis targets: {len(vis_target_set)} unique images.")

        # Collect heatmaps for vis targets by re-running lightweight inference
        img_store_vis: Dict[int, torch.Tensor] = {}
        hm_base_vis:   Dict[int, np.ndarray]   = {}
        hm_comp_vis:   Dict[int, np.ndarray]   = {}

        for ckpt_name, ckpt_path in checkpoints.items():
            model_vis = load_model(ckpt_path, device)
            all_block_indices = list(range(NUM_LAYERS))
            patch_attention_forward(model_vis, all_block_indices, differentiable=False)

            vis_ds = StreamingLocalizationDataset(
                chosen_indices=sorted(vis_target_set & set(chosen_indices)),
                annotations=annotations,
                batch_size=min(args.batch_size, 64),
                num_workers=args.num_workers,
            )

            for batch_cpu, labels, val_indices in vis_ds.iterate_batches():
                B_act = batch_cpu.shape[0]
                batch_gpu = batch_cpu.to(device, non_blocking=True)
                with torch.no_grad():
                    with torch.autocast(device_type=device.type, enabled=args.amp):
                        _ = model_vis(batch_gpu)
                    attn_dict = get_cached_attn_weights(model_vis, all_block_indices)
                    attn_f32  = {k: v.float() for k, v in attn_dict.items()}
                    rollout   = attention_rollout_batch_gpu(attn_f32, device)
                    hms       = upsample_rollout_gpu(rollout)  # (B, 224, 224)

                for i in range(B_act):
                    idx = val_indices[i]
                    if ckpt_name == ckpt_names[0]:   # store image on first pass
                        img_store_vis[idx] = batch_cpu[i].clone()
                        hm_base_vis[idx]   = hms[i].cpu().numpy()
                    else:
                        hm_comp_vis[idx]   = hms[i].cpu().numpy()

            unpatch_attention_forward(model_vis, all_block_indices)
            del model_vis
            torch.cuda.empty_cache()

        # Now save qualitative grids
        save_qualitative_grid(
            vis_dir=output_dir / "visualizations",
            img_store=img_store_vis,
            hm_store_base=hm_base_vis,
            hm_store_comp=hm_comp_vis,
            per_image_base=base_map,
            per_image_comp=comp_map,
            annotations=annotations,
            delta_rows=delta_rows,
            base_name=base_name,
            comp_name=comp_name,
            n_random=n_random,
            n_best=n_best,
            n_worst=n_worst,
            n_highconf=n_highconf,
            seed=args.seed,
        )

    # ── Final table ──
    print("\n" + "="*75)
    print(f"{'Checkpoint':<18} {'Top1':>6} {'PG%':>7} {'Mass%':>7} "
          f"{'mIoU@0.3':>9} {'mIoU@0.5':>9} {'CL@0.3':>7} {'CL@0.5':>7} "
          f"{'img/s':>7}")
    print("="*75)
    for s in all_summaries_csv_rows:
        print(
            f"{s['checkpoint']:<18} "
            f"{s['top1_accuracy']*100:>6.1f} "
            f"{s['pointing_game_accuracy']*100:>7.1f} "
            f"{s['mean_attention_mass_in_box']*100:>7.1f} "
            f"{s.get('mean_iou_thr0.3',0):>9.4f} "
            f"{s.get('mean_iou_thr0.5',0):>9.4f} "
            f"{s.get('corloc_thr0.3',0)*100:>7.1f} "
            f"{s.get('corloc_thr0.5',0)*100:>7.1f} "
            f"{s['throughput_img_s']:>7.1f}"
        )
    print("="*75)

    if len(ckpt_names) >= 2 and len(diffs) > 0:
        print(f"\nDelta ({ckpt_names[1]} − {ckpt_names[0]}) at IoU@0.5:")
        print(f"  Mean:     {delta_summary['mean_delta']:+.5f}  "
              f"95% CI: [{delta_summary['mean_delta_ci95_lo']:+.5f}, "
              f"{delta_summary['mean_delta_ci95_hi']:+.5f}]")
        print(f"  Improved: {delta_summary['improved_pct']:.1f}%  "
              f"Degraded: {delta_summary['degraded_pct']:.1f}%  "
              f"Large gain: {delta_summary['large_gain_pct']:.1f}%  "
              f"Large loss: {delta_summary['large_loss_pct']:.1f}%")

    print(f"\n[DONE] Results → {output_dir.resolve()}")


if __name__ == "__main__":
    main()
