"""CUB-200-2011 attention-based localization evaluation.

Evaluates spatial grounding quality using attention rollout.
Bounding boxes are used ONLY for evaluation, never for training.

Usage:
  python eval_cub_localization.py \
      --checkpoints baseline:runs/cub/baseline_cub_vit_s16/checkpoints/best.pth \
                     token_mild:runs/cub/token_locality_mild_cub_vit_s16/checkpoints/best.pth \
      --hf-dataset bentrevett/caltech-ucsd-birds-200-2011 \
      --batch-size 128 --amp \
      --output-dir runs/cub/cub_token_locality_comparison/
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import timm
from datasets import load_dataset

from common.attention_hooks import (
    patch_attention_forward,
    unpatch_attention_forward,
    get_cached_attn_weights,
)

# ─── Constants ────────────────────────────────────────────────────────────────

IMG_SIZE = 224
GRID = 14
NUM_PATCHES = 196
NUM_LAYERS = 12
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
ATTN_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
CORLOC_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]


# ─── Dataset ─────────────────────────────────────────────────────────────────

def map_bbox_to_224(bbox, orig_w, orig_h):
    """Map bbox from original image coords to 224×224 after Resize(256)+CenterCrop(224).

    Args:
        bbox: [x0, y0, x1, y1] in original pixel coordinates
        orig_w, orig_h: original image dimensions

    Returns:
        (x0, y0, x1, y1) in 224×224 space, or None if degenerate
    """
    x0, y0, x1, y1 = bbox

    # Step 1: Resize shortest side to 256
    scale = 256.0 / min(orig_w, orig_h)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)

    # Scale bbox
    x0s = x0 * scale
    y0s = y0 * scale
    x1s = x1 * scale
    y1s = y1 * scale

    # Step 2: Center crop to 224×224
    cx_offset = (new_w - 224) / 2.0
    cy_offset = (new_h - 224) / 2.0

    x0c = max(0, round(x0s - cx_offset))
    y0c = max(0, round(y0s - cy_offset))
    x1c = min(223, round(x1s - cx_offset))
    y1c = min(223, round(y1s - cy_offset))

    if x1c <= x0c or y1c <= y0c:
        return None  # bbox outside crop area
    return (x0c, y0c, x1c, y1c)


class CUBEvalDataset(Dataset):
    """CUB test set with bbox mapped to 224×224."""

    def __init__(self, hf_split, transform):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        orig_w, orig_h = img.size
        tensor = self.transform(img)
        label = ex["label"]

        bbox_224 = None
        bbox_raw = ex.get("bbox")
        if bbox_raw and len(bbox_raw) == 4:
            bbox_224 = map_bbox_to_224(bbox_raw, orig_w, orig_h)

        return tensor, label, bbox_224, idx


def eval_collate(batch):
    """Custom collate handling None bboxes."""
    tensors = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    bboxes = [b[2] for b in batch]  # list of tuple or None
    indices = [b[3] for b in batch]
    return tensors, labels, bboxes, indices


# ─── Rollout ──────────────────────────────────────────────────────────────────

def attention_rollout_batch(attn_dict, device):
    """Batch attention rollout → (B, 196) on GPU."""
    B = next(iter(attn_dict.values())).shape[0]
    n = NUM_PATCHES + 1  # 197

    rollout = torch.eye(n, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1, -1)
    for layer_idx in sorted(attn_dict.keys()):
        a = attn_dict[layer_idx].float().mean(dim=1)  # (B, n, n)
        eye = torch.eye(n, dtype=a.dtype, device=device).unsqueeze(0)
        a = a + eye
        a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        rollout = torch.bmm(a, rollout)

    cls_row = rollout[:, 0, 1:]  # (B, 196)
    mn = cls_row.min(dim=-1, keepdim=True).values
    mx = cls_row.max(dim=-1, keepdim=True).values
    return (cls_row - mn) / (mx - mn + 1e-8)


def upsample_rollout(grid_maps, img_size=224):
    """(B, 196) → (B, 224, 224)"""
    B = grid_maps.shape[0]
    maps_2d = grid_maps.reshape(B, 1, GRID, GRID)
    up = F.interpolate(maps_2d, size=(img_size, img_size), mode="bilinear", align_corners=False)
    out = up.squeeze(1)
    mn = out.flatten(1).min(dim=1).values[:, None, None]
    mx = out.flatten(1).max(dim=1).values[:, None, None]
    return (out - mn) / (mx - mn + 1e-8)


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_localization_metrics(heatmap, bbox):
    """Single-image localization metrics. heatmap: (224,224) GPU tensor, bbox: (x0,y0,x1,y1)."""
    if bbox is None:
        return None

    x0, y0, x1, y1 = bbox
    results = {}

    # Pointing game
    flat_idx = heatmap.argmax().item()
    py, px = divmod(flat_idx, IMG_SIZE)
    results["pointing_game"] = int(x0 <= px <= x1 and y0 <= py <= y1)

    # Mass in box
    total = heatmap.sum()
    inside = heatmap[y0:y1+1, x0:x1+1].sum()
    results["mass_in_box"] = (inside / total.clamp(min=1e-8)).item()

    # IoU at each threshold
    for t in ATTN_THRESHOLDS:
        binary = (heatmap >= t)
        if not binary.any():
            results[f"iou_thr{t}"] = 0.0
            continue
        rows = binary.any(dim=1).nonzero(as_tuple=False).view(-1)
        cols = binary.any(dim=0).nonzero(as_tuple=False).view(-1)
        if rows.numel() == 0 or cols.numel() == 0:
            results[f"iou_thr{t}"] = 0.0
            continue
        pr_y0, pr_y1 = rows.min().item(), rows.max().item()
        pr_x0, pr_x1 = cols.min().item(), cols.max().item()

        ix0 = max(x0, pr_x0)
        iy0 = max(y0, pr_y0)
        ix1 = min(x1, pr_x1)
        iy1 = min(y1, pr_y1)
        inter = max(0, ix1 - ix0 + 1) * max(0, iy1 - iy0 + 1)
        area_gt = (x1 - x0 + 1) * (y1 - y0 + 1)
        area_pred = (pr_x1 - pr_x0 + 1) * (pr_y1 - pr_y0 + 1)
        union = area_gt + area_pred - inter
        results[f"iou_thr{t}"] = inter / union if union > 0 else 0.0

    return results


# ─── Bootstrap ────────────────────────────────────────────────────────────────

def bootstrap_ci(values, n_resamples=1000, ci=0.95):
    """Bootstrap 95% CI for the mean."""
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    rng = np.random.RandomState(42)
    means = np.array([arr[rng.randint(0, n, n)].mean() for _ in range(n_resamples)])
    lo = float(np.percentile(means, 100 * (1 - ci) / 2))
    hi = float(np.percentile(means, 100 * (1 + ci) / 2))
    return float(arr.mean()), lo, hi


# ─── Main evaluation ─────────────────────────────────────────────────────────

def evaluate_checkpoint(ckpt_name, ckpt_path, eval_loader, device, use_amp):
    """Evaluate one checkpoint on CUB test set."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: {ckpt_name}")
    print(f"{'='*60}")

    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=200)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    all_blocks = list(range(NUM_LAYERS))
    patch_attention_forward(model, all_blocks, differentiable=False)

    per_image = []
    t_start = time.perf_counter()

    for batch_imgs, batch_labels, batch_bboxes, batch_indices in tqdm(
        eval_loader, desc=ckpt_name, unit="batch"
    ):
        B = batch_imgs.shape[0]
        batch_imgs = batch_imgs.to(device, non_blocking=True)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(batch_imgs)
            attn_dict = {k: v.float() for k, v in
                         get_cached_attn_weights(model, all_blocks).items()}
            rollout = attention_rollout_batch(attn_dict, device)
            heatmaps = upsample_rollout(rollout)

        top5 = logits.topk(5, dim=1).indices
        for i in range(B):
            label = batch_labels[i].item()
            pred = top5[i, 0].item()
            top1_ok = int(pred == label)
            top5_ok = int(label in top5[i].tolist())

            bbox = batch_bboxes[i]
            loc_metrics = compute_localization_metrics(heatmaps[i], bbox)

            row = {
                "idx": batch_indices[i],
                "label": label,
                "pred": pred,
                "top1_correct": top1_ok,
                "top5_correct": top5_ok,
            }
            if loc_metrics:
                row.update(loc_metrics)
            per_image.append(row)

    unpatch_attention_forward(model, all_blocks)
    del model
    torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t_start
    n = len(per_image)
    tput = n / elapsed

    # Aggregate
    has_bbox = [r for r in per_image if "pointing_game" in r]
    summary = {
        "checkpoint": ckpt_name,
        "n_images": n,
        "n_with_bbox": len(has_bbox),
        "time_s": round(elapsed, 1),
        "throughput": round(tput, 1),
        "top1": round(100 * np.mean([r["top1_correct"] for r in per_image]), 4),
        "top5": round(100 * np.mean([r["top5_correct"] for r in per_image]), 4),
    }
    if has_bbox:
        summary["pointing_game"] = round(100 * np.mean([r["pointing_game"] for r in has_bbox]), 4)
        summary["mass_in_box"] = round(100 * np.mean([r["mass_in_box"] for r in has_bbox]), 4)
        for t in ATTN_THRESHOLDS:
            key = f"iou_thr{t}"
            summary[f"mean_{key}"] = round(np.mean([r[key] for r in has_bbox]), 6)
        for t in CORLOC_THRESHOLDS:
            vals = [float(r[f"iou_thr{t}"] >= t) for r in has_bbox]
            summary[f"corloc_{t}"] = round(100 * np.mean(vals), 4)

    print(f"  Top1={summary['top1']:.1f}%  Top5={summary['top5']:.1f}%  "
          f"PG={summary.get('pointing_game', 0):.1f}%  "
          f"Mass={summary.get('mass_in_box', 0):.1f}%  "
          f"CL@0.5={summary.get('corloc_0.5', 0):.1f}%  "
          f"{tput:.0f} img/s")

    return per_image, summary


# ─── Visualization ────────────────────────────────────────────────────────────

def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (tensor.cpu().float() * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def save_debug_visualizations(model, eval_loader, device, use_amp, output_dir, n=20):
    """Save n debug visualizations to verify bbox mapping + rollout."""
    debug_dir = Path(output_dir) / "debug_vis"
    debug_dir.mkdir(parents=True, exist_ok=True)

    all_blocks = list(range(NUM_LAYERS))
    patch_attention_forward(model, all_blocks, differentiable=False)

    saved = 0
    for batch_imgs, batch_labels, batch_bboxes, batch_indices in eval_loader:
        B = batch_imgs.shape[0]
        batch_imgs_gpu = batch_imgs.to(device, non_blocking=True)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp):
                _ = model(batch_imgs_gpu)
            attn_dict = {k: v.float() for k, v in
                         get_cached_attn_weights(model, all_blocks).items()}
            rollout = attention_rollout_batch(attn_dict, device)
            heatmaps = upsample_rollout(rollout)

        for i in range(B):
            if saved >= n:
                break
            bbox = batch_bboxes[i]
            if bbox is None:
                continue

            img_np = denormalize(batch_imgs[i])
            hm_np = heatmaps[i].cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_np)
            rect = mpatches.Rectangle(
                (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                linewidth=2, edgecolor="lime", facecolor="none"
            )
            axes[0].add_patch(rect)
            axes[0].set_title(f"GT bbox (label={batch_labels[i].item()})", fontsize=8)
            axes[0].axis("off")

            axes[1].imshow(img_np)
            axes[1].imshow(hm_np, alpha=0.55, cmap="jet", vmin=0, vmax=1)
            axes[1].set_title("Rollout overlay", fontsize=8)
            axes[1].axis("off")

            axes[2].imshow(hm_np, cmap="jet", vmin=0, vmax=1)
            axes[2].set_title("Rollout heatmap", fontsize=8)
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(debug_dir / f"debug_{saved:04d}.png", dpi=90, bbox_inches="tight")
            plt.close()
            saved += 1

        if saved >= n:
            break

    unpatch_attention_forward(model, all_blocks)
    print(f"[INFO] Saved {saved} debug visualizations → {debug_dir}")


# ─── Plots ────────────────────────────────────────────────────────────────────

def make_comparison_plots(summaries, per_image_all, output_dir):
    """Generate comparison plots across checkpoints."""
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    names = list(summaries.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    # 1. Top1 bar
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [summaries[n]["top1"] for n in names]
    ax.bar(names, vals, color=colors)
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("CUB-200 Classification")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.2, f"{v:.1f}%", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "top1_bar.png", dpi=120)
    plt.close()

    # 2. Pointing game bar
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [summaries[n].get("pointing_game", 0) for n in names]
    ax.bar(names, vals, color=colors)
    ax.set_ylabel("Pointing Game (%)")
    ax.set_title("Pointing Game Accuracy")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.2, f"{v:.1f}%", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "pointing_game_bar.png", dpi=120)
    plt.close()

    # 3. CorLoc curve
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, name in enumerate(names):
        corloc_vals = [summaries[name].get(f"corloc_{t}", 0) for t in CORLOC_THRESHOLDS]
        ax.plot(CORLOC_THRESHOLDS, corloc_vals, marker="o", label=name, color=colors[i])
    ax.set_xlabel("IoU Threshold")
    ax.set_ylabel("CorLoc (%)")
    ax.set_title("CorLoc Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "corloc_curve.png", dpi=120)
    plt.close()

    # 4. mIoU curve
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, name in enumerate(names):
        miou_vals = [summaries[name].get(f"mean_iou_thr{t}", 0) for t in ATTN_THRESHOLDS]
        ax.plot(ATTN_THRESHOLDS, miou_vals, marker="o", label=name, color=colors[i])
    ax.set_xlabel("Attention Threshold")
    ax.set_ylabel("Mean IoU")
    ax.set_title("mIoU vs Attention Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "miou_curve.png", dpi=120)
    plt.close()

    # 5. Mass in box bar
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [summaries[n].get("mass_in_box", 0) for n in names]
    ax.bar(names, vals, color=colors)
    ax.set_ylabel("Attention Mass in Bbox (%)")
    ax.set_title("Attention Mass Inside GT Bounding Box")
    plt.tight_layout()
    plt.savefig(plots_dir / "mass_in_box_bar.png", dpi=120)
    plt.close()

    print(f"[INFO] Plots saved → {plots_dir}")


# ─── Delta analysis ──────────────────────────────────────────────────────────

def compute_deltas(base_rows, comp_rows, base_name, comp_name, output_dir):
    """Per-image IoU delta at threshold 0.3."""
    base_map = {r["idx"]: r for r in base_rows if "iou_thr0.3" in r}
    comp_map = {r["idx"]: r for r in comp_rows if "iou_thr0.3" in r}
    common = sorted(set(base_map) & set(comp_map))

    delta_rows = []
    diffs = []
    for idx in common:
        b_iou = float(base_map[idx]["iou_thr0.3"])
        c_iou = float(comp_map[idx]["iou_thr0.3"])
        d = c_iou - b_iou
        diffs.append(d)
        delta_rows.append({
            "idx": idx,
            f"{base_name}_iou03": round(b_iou, 6),
            f"{comp_name}_iou03": round(c_iou, 6),
            "delta": round(d, 6),
        })

    out_path = Path(output_dir) / f"per_image_delta_{comp_name}_vs_{base_name}.csv"
    if delta_rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(delta_rows[0].keys()))
            writer.writeheader()
            writer.writerows(delta_rows)
        print(f"[INFO] Delta CSV → {out_path}")

    return np.array(diffs)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CUB-200 localization evaluation")
    parser.add_argument("--checkpoints", nargs="+", required=True, metavar="NAME:PATH")
    parser.add_argument("--hf-dataset", default="bentrevett/caltech-ucsd-birds-200-2011")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-dir", default="runs/cub/cub_token_locality_comparison/")
    parser.add_argument("--debug-vis", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Parse checkpoints
    checkpoints = {}
    for entry in args.checkpoints:
        if ":" not in entry:
            print(f"[ERROR] Format must be 'name:path', got: {entry}")
            sys.exit(1)
        name, path = entry.split(":", 1)
        if not os.path.exists(path):
            print(f"[WARNING] Checkpoint not found, skipping: {path}")
            continue
        checkpoints[name] = path

    if not checkpoints:
        print("[ERROR] No valid checkpoints found.")
        sys.exit(1)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Checkpoints: {list(checkpoints.keys())}")
    print(f"[INFO] GPU: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}")

    # Load dataset
    print(f"[INFO] Loading CUB dataset: {args.hf_dataset}")
    ds = load_dataset(args.hf_dataset)
    test_split = ds.get("test", ds.get("validation"))
    if test_split is None:
        print("[ERROR] No test split found.")
        sys.exit(1)

    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_ds = CUBEvalDataset(test_split, val_transform)
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=eval_collate,
    )
    print(f"[INFO] Test set: {len(eval_ds)} images")

    # Debug visualizations (first checkpoint)
    if args.debug_vis > 0:
        first_path = list(checkpoints.values())[0]
        dbg_model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=200)
        ckpt = torch.load(first_path, map_location=device, weights_only=False)
        dbg_model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        dbg_model = dbg_model.to(device).eval()
        save_debug_visualizations(dbg_model, eval_loader, device, args.amp, output_dir, args.debug_vis)
        del dbg_model
        torch.cuda.empty_cache()

    # Evaluate each checkpoint
    all_per_image = {}
    all_summaries = {}
    for ckpt_name, ckpt_path in checkpoints.items():
        per_image, summary = evaluate_checkpoint(
            ckpt_name, ckpt_path, eval_loader, device, args.amp
        )
        all_per_image[ckpt_name] = per_image
        all_summaries[ckpt_name] = summary

        # Save per-image CSV
        csv_path = output_dir / f"per_image_results_{ckpt_name}.csv"
        if per_image:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
                writer.writeheader()
                writer.writerows(per_image)

    # Save metrics summary
    summary_path = output_dir / "metrics_summary.csv"
    if all_summaries:
        rows = list(all_summaries.values())
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[INFO] Metrics summary → {summary_path}")

    # Bootstrap CIs
    ckpt_names = list(checkpoints.keys())
    ci_results = {}
    for name in ckpt_names:
        has_bbox = [r for r in all_per_image[name] if "pointing_game" in r]
        if not has_bbox:
            continue
        ci = {}
        for metric in ["pointing_game", "mass_in_box", "iou_thr0.3", "iou_thr0.5"]:
            vals = [r[metric] for r in has_bbox]
            pt, lo, hi = bootstrap_ci(vals)
            ci[metric] = {"point": round(pt, 6), "ci95_lo": round(lo, 6), "ci95_hi": round(hi, 6)}
        ci_results[name] = ci

    ci_path = output_dir / "bootstrap_confidence_intervals.json"
    with open(ci_path, "w") as f:
        json.dump(ci_results, f, indent=2)

    # Delta analysis (vs first checkpoint = baseline)
    if len(ckpt_names) >= 2:
        base_name = ckpt_names[0]
        for comp_name in ckpt_names[1:]:
            diffs = compute_deltas(
                all_per_image[base_name], all_per_image[comp_name],
                base_name, comp_name, output_dir
            )
            if len(diffs) > 0:
                pt, lo, hi = bootstrap_ci(diffs.tolist())
                print(f"  Delta ({comp_name} − {base_name}) IoU@0.3: "
                      f"mean={pt:+.4f} CI=[{lo:+.4f}, {hi:+.4f}]")

    # Plots
    make_comparison_plots(all_summaries, all_per_image, output_dir)

    # Print decision summary
    print(f"\n{'='*70}")
    print("LOCALIZATION EVALUATION SUMMARY")
    print(f"{'='*70}")
    header = f"{'Checkpoint':<25} {'Top1':>6} {'PG':>6} {'Mass':>6} {'CL@0.3':>7} {'CL@0.5':>7}"
    print(header)
    print("-" * len(header))
    for name in ckpt_names:
        s = all_summaries[name]
        print(f"{name:<25} {s['top1']:>6.1f} {s.get('pointing_game',0):>6.1f} "
              f"{s.get('mass_in_box',0):>6.1f} {s.get('corloc_0.3',0):>7.1f} "
              f"{s.get('corloc_0.5',0):>7.1f}")

    # Success criteria check
    if len(ckpt_names) >= 2:
        base_s = all_summaries[ckpt_names[0]]
        print(f"\nSuccess criteria (vs {ckpt_names[0]}):")
        for comp_name in ckpt_names[1:]:
            comp_s = all_summaries[comp_name]
            top1_drop = base_s["top1"] - comp_s["top1"]
            pg_gain = comp_s.get("pointing_game", 0) - base_s.get("pointing_game", 0)
            cl3_gain = comp_s.get("corloc_0.3", 0) - base_s.get("corloc_0.3", 0)
            cl5_gain = comp_s.get("corloc_0.5", 0) - base_s.get("corloc_0.5", 0)
            iou5_gain = comp_s.get("mean_iou_thr0.5", 0) - base_s.get("mean_iou_thr0.5", 0)

            interesting = (top1_drop <= 0.5 and pg_gain >= 2.0 and
                           cl3_gain >= 2.0 and cl5_gain >= 1.0)
            very_interesting = (top1_drop <= 0 and pg_gain >= 4.0 and
                                cl3_gain >= 4.0 and cl5_gain >= 3.0)

            print(f"\n  [{comp_name}]")
            print(f"    Top1 drop:  {top1_drop:+.2f}pp  {'✓' if top1_drop <= 0.5 else '✗'}")
            print(f"    PG gain:    {pg_gain:+.2f}pp  {'✓' if pg_gain >= 2.0 else '✗'}")
            print(f"    CL@0.3:     {cl3_gain:+.2f}pp  {'✓' if cl3_gain >= 2.0 else '✗'}")
            print(f"    CL@0.5:     {cl5_gain:+.2f}pp  {'✓' if cl5_gain >= 1.0 else '✗'}")
            print(f"    mIoU@0.5:   {iou5_gain:+.5f}  {'✓' if iou5_gain >= 0.05 else '✗'}")
            print(f"    → {'VERY INTERESTING' if very_interesting else 'INTERESTING' if interesting else 'Not yet compelling'}")

    print(f"\n[DONE] Results → {output_dir.resolve()}")


if __name__ == "__main__":
    main()
