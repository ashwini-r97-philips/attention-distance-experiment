"""Evaluation: accuracy, attention stats, GMM bimodality, summary table.

Usage:
  python evaluate.py --config configs/baseline.yaml \
      --checkpoint runs/baseline/checkpoints/best.pth
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from common.config import load_config
from common.model_utils import load_vit_small
from common.attention_hooks import capture_attention
from common.mad_metrics import (
    build_distance_matrix,
    compute_mad,
    compute_local_mass,
    compute_attention_entropy,
    compute_inter_head_mad_variance,
    compute_distance_histogram,
    compute_head_correlation,
)
from data import get_val_loader, get_attention_eval_subset


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


@torch.no_grad()
def evaluate_accuracy(model, val_loader, device):
    model.eval()
    correct1 = correct5 = total = 0
    for images, targets in tqdm(val_loader, desc="Accuracy", leave=False):
        images, targets = images.to(device), targets.to(device)
        out = model(images)
        _, pred5 = out.topk(5, dim=1)
        correct1 += (pred5[:, 0] == targets).sum().item()
        correct5 += (pred5 == targets.unsqueeze(1)).any(dim=1).sum().item()
        total += targets.size(0)
    return {
        "top1": round(100.0 * correct1 / max(1, total), 2),
        "top5": round(100.0 * correct5 / max(1, total), 2),
    }


@torch.no_grad()
def compute_full_attention_stats(model, loader, device, dist_matrix, cfg):
    """Compute MAD, local mass, entropy, distance histograms per layer/head."""
    model.eval()
    all_blocks = list(range(cfg.num_blocks))
    accum = {b: {"mad": [], "entropy": [], "dist_hist": []} for b in all_blocks}
    for tau in cfg.tau_values:
        for b in all_blocks:
            accum[b][f"lm_{tau}"] = []

    for images, _ in tqdm(loader, desc="Attn stats", leave=False):
        images = images.to(device)
        with capture_attention(model, all_blocks) as get_attn:
            _ = model(images)
            attn_dict = get_attn()
        for b in all_blocks:
            a = attn_dict[b]
            accum[b]["mad"].append(compute_mad(a, dist_matrix).cpu().numpy())
            accum[b]["entropy"].append(compute_attention_entropy(a).cpu().numpy())
            accum[b]["dist_hist"].append(compute_distance_histogram(a, dist_matrix).cpu().numpy())
            for tau in cfg.tau_values:
                accum[b][f"lm_{tau}"].append(compute_local_mass(a, dist_matrix, tau=tau).cpu().numpy())

    stats = {}
    for b in all_blocks:
        stats[b] = {}
        for key in accum[b]:
            arr = np.stack(accum[b][key], axis=0)
            stats[b][key] = np.mean(arr, axis=0).tolist()
    return stats


@torch.no_grad()
def gmm_bimodality_test(model, loader, device, dist_matrix, cfg):
    """Fit 1- and 2-component GMM to per-head MAD per layer."""
    model.eval()
    all_blocks = list(range(cfg.num_blocks))
    mad_per_batch = {b: [] for b in all_blocks}

    for images, _ in tqdm(loader, desc="GMM data", leave=False):
        images = images.to(device)
        with capture_attention(model, all_blocks) as get_attn:
            _ = model(images)
            attn_dict = get_attn()
        for b in all_blocks:
            mad_per_batch[b].append(compute_mad(attn_dict[b], dist_matrix).cpu().numpy())

    results = {}
    for b in all_blocks:
        all_mads = np.stack(mad_per_batch[b], axis=0)
        flat = all_mads.flatten().reshape(-1, 1)
        gmm1 = GaussianMixture(n_components=1, random_state=42).fit(flat)
        gmm2 = GaussianMixture(n_components=2, random_state=42).fit(flat)
        results[b] = {
            "aic_1comp": float(gmm1.aic(flat)),
            "aic_2comp": float(gmm2.aic(flat)),
            "bic_1comp": float(gmm1.bic(flat)),
            "bic_2comp": float(gmm2.bic(flat)),
            "bic_diff": float(gmm1.bic(flat) - gmm2.bic(flat)),
            "gmm2_means": gmm2.means_.flatten().tolist(),
            "gmm2_weights": gmm2.weights_.tolist(),
            "per_head_mean_mad": all_mads.mean(axis=0).tolist(),
        }
    return results


def aggregate_summary(accuracy, attn_stats, cfg):
    """Build summary dict for the summary table."""
    all_mads = []
    all_vars = []
    all_ent = []
    lm_agg = {tau: [] for tau in cfg.tau_values}

    for b in range(cfg.num_blocks):
        mads = np.array(attn_stats[b]["mad"])
        all_mads.extend(mads.tolist())
        all_vars.append(float(np.var(mads)))
        all_ent.extend(np.array(attn_stats[b]["entropy"]).tolist())
        for tau in cfg.tau_values:
            lm_agg[tau].extend(np.array(attn_stats[b][f"lm_{tau}"]).tolist())

    summary = {
        "top1": accuracy["top1"],
        "top5": accuracy["top5"],
        "mean_mad": round(float(np.mean(all_mads)), 4),
        "mean_mad_variance": round(float(np.mean(all_vars)), 6),
        "mean_entropy": round(float(np.mean(all_ent)), 4),
    }
    for tau in cfg.tau_values:
        summary[f"mean_local_mass_{tau}"] = round(float(np.mean(lm_agg[tau])), 4)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = cfg.device
    output_dir = args.output_dir or os.path.join(cfg.output_dir, "eval")
    os.makedirs(output_dir, exist_ok=True)

    model = load_vit_small(cfg, pretrained=False)
    model = load_checkpoint(model, args.checkpoint, device)
    dist_matrix = build_distance_matrix(cfg.grid_h, cfg.grid_w, device=device)

    val_loader = get_val_loader(cfg, batch_size=256)
    attn_loader, _ = get_attention_eval_subset(cfg)

    print("Evaluating accuracy...")
    accuracy = evaluate_accuracy(model, val_loader, device)
    print(f"  Top-1: {accuracy['top1']}%  Top-5: {accuracy['top5']}%")

    print("Computing attention stats...")
    attn_stats = compute_full_attention_stats(model, attn_loader, device, dist_matrix, cfg)

    print("Running GMM bimodality test...")
    gmm = gmm_bimodality_test(model, attn_loader, device, dist_matrix, cfg)

    summary = aggregate_summary(accuracy, attn_stats, cfg)

    results = {
        "accuracy": accuracy,
        "attention_stats": {str(k): v for k, v in attn_stats.items()},
        "gmm": {str(k): v for k, v in gmm.items()},
        "summary": summary,
    }

    out_path = os.path.join(output_dir, "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    # Print summary
    print("\n" + "=" * 40)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("=" * 40)


if __name__ == "__main__":
    main()
