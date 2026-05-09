"""Baseline analysis: measure attention metrics on pretrained DeiT-S (no training).

Generates MAD heatmap, violin plots, and sample attention maps.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from common import config as cfg
from common.model_utils import load_deit_small
from common.attention_hooks import capture_attention
from common.mad_metrics import build_distance_matrix, compute_mad, compute_non_self_mad, compute_local_mass, compute_attention_entropy
from common.plot_utils import plot_mad_heatmap, plot_mad_violins
from data import get_val_loader


def run_baseline_analysis(data_root=None, num_batches=None, batch_size=64, output_dir=None):
    device = cfg.DEVICE
    output_dir = output_dir or cfg.BASELINE_ANALYSIS_DIR
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    num_batches = num_batches or cfg.NUM_EVAL_BATCHES

    print(f"Loading pretrained {cfg.MODEL_NAME}...")
    model = load_deit_small(pretrained=True, device=device)
    model.eval()

    print(f"Loading validation data from {data_root or cfg.DATA_ROOT}...")
    val_loader = get_val_loader(data_root=data_root, batch_size=batch_size)

    dist_matrix = build_distance_matrix(device=device)
    all_blocks = list(range(cfg.NUM_BLOCKS))

    # Accumulators: per-block, per-head
    mad_accum = {b: [] for b in all_blocks}
    non_self_mad_accum = {b: [] for b in all_blocks}
    local_mass_accum = {b: [] for b in all_blocks}
    entropy_accum = {b: [] for b in all_blocks}

    print(f"Running analysis over {num_batches} batches...")
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(val_loader, total=num_batches)):
            if batch_idx >= num_batches:
                break

            images = images.to(device)
            with capture_attention(model, all_blocks) as get_attn:
                _ = model(images)
                attn_dict = get_attn()

            for bidx in all_blocks:
                attn = attn_dict[bidx]  # (B, H, N, N)
                mad_accum[bidx].append(compute_mad(attn, dist_matrix).cpu().numpy())
                non_self_mad_accum[bidx].append(compute_non_self_mad(attn, dist_matrix).cpu().numpy())
                local_mass_accum[bidx].append(compute_local_mass(attn, dist_matrix).cpu().numpy())
                entropy_accum[bidx].append(compute_attention_entropy(attn).cpu().numpy())

    # Average across batches
    mad_avg = {b: np.mean(mad_accum[b], axis=0) for b in all_blocks}
    non_self_mad_avg = {b: np.mean(non_self_mad_accum[b], axis=0) for b in all_blocks}
    local_mass_avg = {b: np.mean(local_mass_accum[b], axis=0) for b in all_blocks}
    entropy_avg = {b: np.mean(entropy_accum[b], axis=0) for b in all_blocks}

    # Per-batch MAD for violin plots
    mad_per_batch = {b: np.stack(mad_accum[b], axis=0) for b in all_blocks}

    # Save metrics
    metrics = {}
    for b in all_blocks:
        metrics[str(b)] = {
            "mad": mad_avg[b].tolist(),
            "non_self_mad": non_self_mad_avg[b].tolist(),
            "local_mass": local_mass_avg[b].tolist(),
            "entropy": entropy_avg[b].tolist(),
        }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Generate plots
    plot_mad_heatmap(mad_avg, os.path.join(output_dir, "plots", "mad_heatmap.png"),
                     title="Pretrained DeiT-S: Mean Attention Distance")
    plot_mad_heatmap(non_self_mad_avg, os.path.join(output_dir, "plots", "non_self_mad_heatmap.png"),
                     title="Pretrained DeiT-S: Non-Self MAD")
    plot_mad_heatmap(local_mass_avg, os.path.join(output_dir, "plots", "local_mass_heatmap.png"),
                     title="Pretrained DeiT-S: Local Mass (τ={})".format(cfg.LOCAL_RADIUS_TAU))
    plot_mad_heatmap(entropy_avg, os.path.join(output_dir, "plots", "entropy_heatmap.png"),
                     title="Pretrained DeiT-S: Attention Entropy")

    # Violin plots for early blocks
    early_blocks = cfg.REGULARIZED_BLOCKS
    plot_mad_violins(
        {b: mad_per_batch[b] for b in early_blocks},
        os.path.join(output_dir, "plots", "mad_violins_early.png"),
        block_indices=early_blocks,
        title="Pretrained DeiT-S: MAD Distributions (Early Blocks)"
    )
    plot_mad_violins(
        mad_per_batch,
        os.path.join(output_dir, "plots", "mad_violins_all.png"),
        title="Pretrained DeiT-S: MAD Distributions (All Blocks)"
    )

    print(f"Plots saved to {os.path.join(output_dir, 'plots')}")
    print("\n=== Summary ===")
    for b in all_blocks:
        mads = mad_avg[b]
        print(f"Block {b:2d}: MAD = [{', '.join(f'{v:.4f}' for v in mads)}]  "
              f"range={mads.max()-mads.min():.4f}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--num_batches", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    run_baseline_analysis(
        data_root=args.data_root,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
