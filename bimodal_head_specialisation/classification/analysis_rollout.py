"""Attention rollout, attention flow, and GMAR (Gradient-weighted Multi-head
Attention Rollout) analysis.

References:
  - Abnar & Zuidema, "Quantifying Attention Flow in Transformers", ACL 2020
    (arXiv:2005.00928)
  - GMAR, arXiv:2504.19414, ICASSP 2025 — rollout weighted by gradient signal

Usage:
  python analysis_rollout.py \
      --config configs/baseline.yaml \
      --baseline_ckpt runs/baseline/checkpoints/best.pth \
      [--targeted_ckpt runs/spread_weak/checkpoints/best.pth] \
      --output_dir analysis/rollout/

Outputs:
  - rollout_baseline.json/png        Attention rollout heatmaps (CLS → patches)
  - flow_baseline.json/png           Attention flow (max-flow formulation)
  - gmar_baseline.json/png           Gradient-weighted rollout
  - Per-head contribution maps for selected images
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from common.config import load_config
from common.model_utils import load_vit_small
from common.attention_hooks import patch_attention_forward, get_cached_attn_weights, unpatch_attention_forward
from data import get_attention_eval_subset


# ─── Rollout ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def attention_rollout(attn_list, head_reduction="mean", discard_ratio=0.0):
    """Compute attention rollout across layers.

    Args:
        attn_list: list of (B, H, N, N) tensors per layer (ordered 0..L-1).
        head_reduction: "mean" averages heads; "max" takes max over heads.
        discard_ratio: fraction of lowest-attention entries to zero out per layer
                       before rollout (improves signal-to-noise).

    Returns:
        rollout: (B, N, N) — effective attention from each token to every other.
    """
    result = None
    for attn in attn_list:
        if head_reduction == "mean":
            attn_heads = attn.mean(dim=1)  # (B, N, N)
        elif head_reduction == "max":
            attn_heads = attn.max(dim=1).values
        else:
            raise ValueError(f"Unknown head_reduction: {head_reduction}")

        if discard_ratio > 0:
            B, N, _ = attn_heads.shape
            flat = attn_heads.reshape(B, -1)
            k = int(flat.shape[1] * discard_ratio)
            if k > 0:
                thresh = flat.kthvalue(k, dim=1).values.unsqueeze(-1)
                attn_heads = attn_heads * (flat.reshape(B, N, N) > thresh).float()
                # Renormalize
                attn_heads = attn_heads / attn_heads.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Add residual connection (identity)
        I = torch.eye(attn_heads.shape[-1], device=attn_heads.device).unsqueeze(0)
        attn_with_res = 0.5 * attn_heads + 0.5 * I

        # Renormalize rows
        attn_with_res = attn_with_res / attn_with_res.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        if result is None:
            result = attn_with_res
        else:
            result = attn_with_res @ result

    return result


# ─── Attention Flow ───────────────────────────────────────────────────────────

@torch.no_grad()
def attention_flow(attn_list, head_reduction="mean"):
    """Attention flow via max-flow / DAG formulation (Abnar & Zuidema 2020).

    Approximate the maximum flow from input to output through the attention
    graph by computing joint attention across layers as the element-wise
    maximum of all paths, computed by iterative matrix multiplication with
    max over intermediate nodes.

    For computational tractability, we use the iterative joint-attention
    formulation: A^{flow} = A_L * ... * A_1 where * is max-based composition.

    This is implemented as the product of stochastic matrices
    (same as rollout but without residual mixing, keeping raw attention).

    Returns:
        flow: (B, N, N) effective attention flow.
    """
    result = None
    for attn in attn_list:
        if head_reduction == "mean":
            a = attn.mean(dim=1)
        else:
            a = attn.max(dim=1).values

        if result is None:
            result = a
        else:
            result = a @ result
    return result


# ─── GMAR (Gradient-weighted Multi-head Attention Rollout) ────────────────────

def gmar(model, images, target_class=None, device="cpu"):
    """Gradient-weighted multi-head attention rollout.

    Weights each head's attention by the gradient of the target class logit
    w.r.t. the attention weights, then performs rollout with these weighted maps.

    Args:
        model: ViT with attention hooks patched (differentiable mode).
        images: (B, C, H, W) input tensor.
        target_class: int or None (uses predicted class if None).

    Returns:
        gmar_map: (B, N_patches) — relevance of each patch to the prediction.
        per_head_grad_weight: dict {layer: (H,)} gradient importance per head.
    """
    model.eval()
    num_blocks = len(model.blocks)
    all_blocks = list(range(num_blocks))

    # Patch with differentiable attention
    patch_attention_forward(model, all_blocks, differentiable=True)

    try:
        images = images.to(device).requires_grad_(False)
        out = model(images)  # (B, num_classes)

        if target_class is None:
            target_class = out.argmax(dim=-1)  # (B,)
        elif isinstance(target_class, int):
            target_class = torch.full((images.shape[0],), target_class,
                                      device=device, dtype=torch.long)

        # Gather target logits and backprop
        target_logits = out.gather(1, target_class.unsqueeze(1)).sum()

        model.zero_grad()
        target_logits.backward(retain_graph=False)

        # Collect attention weights and their gradients
        attn_weights = get_cached_attn_weights(model, all_blocks)
        grad_weighted_attns = []
        per_head_grad_weight = {}

        for b in all_blocks:
            attn = attn_weights[b]  # (B, H, N, N)
            if attn.grad is not None:
                grad = attn.grad  # (B, H, N, N)
            else:
                # Fallback: gradient may not reach this layer
                grad = torch.ones_like(attn)

            # Per-head gradient importance: mean absolute gradient
            head_importance = grad.abs().mean(dim=(0, 2, 3))  # (H,)
            per_head_grad_weight[b] = head_importance.detach().cpu().numpy().tolist()

            # Weight attention by gradient magnitude, then average over heads
            weighted = (attn.detach() * grad.detach().abs())
            weighted_mean = weighted.mean(dim=1)  # (B, N, N)

            # Normalize rows
            weighted_mean = weighted_mean / weighted_mean.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            grad_weighted_attns.append(weighted_mean)

        # Rollout with grad-weighted attention
        result = None
        for gw_attn in grad_weighted_attns:
            I = torch.eye(gw_attn.shape[-1], device=device).unsqueeze(0)
            a = 0.5 * gw_attn + 0.5 * I
            a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            if result is None:
                result = a
            else:
                result = a @ result

        # Extract CLS row → patch relevance (skip CLS-to-CLS)
        gmar_map = result[:, 0, 1:]  # (B, N_patches)
        gmar_map = gmar_map / gmar_map.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return gmar_map.detach(), per_head_grad_weight

    finally:
        unpatch_attention_forward(model, all_blocks)


# ─── Collect attention maps for rollout/flow ──────────────────────────────────

@torch.no_grad()
def collect_attention_maps(model, images, device):
    """Forward pass collecting all layers' attention weights."""
    model.eval()
    all_blocks = list(range(len(model.blocks)))
    patch_attention_forward(model, all_blocks, differentiable=False)
    try:
        images = images.to(device)
        _ = model(images)
        attn_dict = get_cached_attn_weights(model, all_blocks)
        # Return as ordered list
        return [attn_dict[b] for b in all_blocks]
    finally:
        unpatch_attention_forward(model, all_blocks)


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_rollout_grid(maps, grid_h, grid_w, save_path, title="Attention Rollout",
                      num_images=8):
    """Plot CLS→patch rollout maps for a batch of images."""
    n = min(num_images, maps.shape[0])
    fig, axes = plt.subplots(2, n // 2, figsize=(3 * (n // 2), 6))
    axes = axes.flatten()
    for i in range(n):
        m = maps[i].reshape(grid_h, grid_w)
        axes[i].imshow(m, cmap="inferno", interpolation="bilinear")
        axes[i].set_title(f"Image {i}", fontsize=8)
        axes[i].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_gmar_head_importance(head_weights, save_path, title="GMAR Head Importance"):
    """Plot per-head gradient importance across layers."""
    layers = sorted(head_weights.keys())
    num_heads = len(head_weights[layers[0]])
    mat = np.array([head_weights[l] for l in layers])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns_available = True
    try:
        import seaborn as sns
        sns.heatmap(mat, ax=ax, cmap="YlOrRd", annot=True, fmt=".3f",
                    xticklabels=[f"H{h}" for h in range(num_heads)],
                    yticklabels=[f"L{l}" for l in layers])
    except ImportError:
        ax.imshow(mat, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(num_heads))
        ax.set_yticks(range(len(layers)))
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Attention rollout, flow & GMAR")
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline_ckpt", required=True)
    parser.add_argument("--targeted_ckpt", default=None)
    parser.add_argument("--output_dir", default="analysis/rollout")
    parser.add_argument("--num_images", type=int, default=8,
                        help="Number of images to visualise rollout/GMAR maps for.")
    parser.add_argument("--discard_ratio", type=float, default=0.1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = cfg.device
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    loader, _ = get_attention_eval_subset(cfg)

    def _run_for_model(ckpt_path, tag):
        print(f"\n{'='*40}")
        print(f"  Processing: {tag}")
        print(f"{'='*40}")

        model = load_vit_small(cfg, pretrained=False)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        # Grab a single batch for visualisation
        vis_batch = next(iter(loader))
        images, labels = vis_batch
        images = images[:args.num_images]

        # ── Rollout ──
        print("Computing attention rollout...")
        attn_list = collect_attention_maps(model, images, device)
        rollout_map = attention_rollout(attn_list, discard_ratio=args.discard_ratio)
        cls_rollout = rollout_map[:, 0, 1:]  # (B, N_patches)
        cls_rollout_np = cls_rollout.cpu().numpy()

        with open(os.path.join(out, f"rollout_{tag}.json"), "w") as f:
            json.dump({"cls_to_patches": cls_rollout_np.tolist()}, f)
        plot_rollout_grid(cls_rollout_np, cfg.grid_h, cfg.grid_w,
                          os.path.join(out, f"rollout_{tag}.png"),
                          title=f"Attention Rollout ({tag})")

        # ── Flow ──
        print("Computing attention flow...")
        flow_map = attention_flow(attn_list)
        cls_flow = flow_map[:, 0, 1:]
        cls_flow_np = cls_flow.cpu().numpy()

        with open(os.path.join(out, f"flow_{tag}.json"), "w") as f:
            json.dump({"cls_to_patches": cls_flow_np.tolist()}, f)
        plot_rollout_grid(cls_flow_np, cfg.grid_h, cfg.grid_w,
                          os.path.join(out, f"flow_{tag}.png"),
                          title=f"Attention Flow ({tag})")

        # ── GMAR ──
        print("Computing GMAR...")
        gmar_map, head_weights = gmar(model, images, device=device)
        gmar_np = gmar_map.cpu().numpy()

        with open(os.path.join(out, f"gmar_{tag}.json"), "w") as f:
            json.dump({
                "gmar_maps": gmar_np.tolist(),
                "head_gradient_importance": {str(k): v for k, v in head_weights.items()},
            }, f, indent=2)
        plot_rollout_grid(gmar_np, cfg.grid_h, cfg.grid_w,
                          os.path.join(out, f"gmar_{tag}.png"),
                          title=f"GMAR ({tag})")
        plot_gmar_head_importance(
            head_weights, os.path.join(out, f"gmar_head_importance_{tag}.png"),
            title=f"GMAR Head Gradient Importance ({tag})")

        del model
        torch.cuda.empty_cache()

    _run_for_model(args.baseline_ckpt, "baseline")
    if args.targeted_ckpt:
        _run_for_model(args.targeted_ckpt, "targeted")

    print("\nRollout / Flow / GMAR analysis complete.")


if __name__ == "__main__":
    main()
