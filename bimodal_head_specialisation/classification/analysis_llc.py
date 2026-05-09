"""Refined Local Learning Coefficient (LLC) per attention head.

Computes a per-head complexity measure grounded in Singular Learning Theory
(SLT). A higher LLC indicates a head occupying a more complex (higher-
codimension) region of the loss landscape — interpretable as more specialised.

Reference: Hoogland et al., "The Developmental Landscape of In-Context
Learning", arXiv:2410.02984.

Method:
  1. For each head, define a per-head loss: freeze all params except those
     in the head's Q, K, V, and output projection rows.
  2. Estimate the LLC via SGLD (Stochastic Gradient Langevin Dynamics)
     sampling around the converged parameters:
       LLC_h ≈ (1/n) * [E_β[L(θ)] - L(θ*)]
     where β = inverse temperature, θ* = converged params, and the
     expectation is over the SGLD posterior.
  3. The ratio LLC_h / (d_h / 2) gives the normalised learning coefficient
     λ̂_h, where d_h is the parameter count of head h.

Usage:
  python analysis_llc.py \
      --config configs/baseline.yaml \
      --checkpoint runs/baseline/checkpoints/best.pth \
      --output_dir analysis/llc/
      [--sgld_steps 500] [--sgld_lr 1e-5] [--num_chains 3]

Outputs:
  - llc_per_head.json         Raw LLC values per layer × head
  - llc_heatmap.png           (num_blocks × num_heads) heatmap
  - llc_vs_mad.png            Scatter: LLC vs MAD per head
"""

import argparse
import json
import os
import sys
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from common.config import load_config
from common.model_utils import load_vit_small
from common.attention_hooks import capture_attention
from common.mad_metrics import build_distance_matrix, compute_mad
from data import get_attention_eval_subset


# ─── Head parameter identification ───────────────────────────────────────────

def get_head_params(model, block_idx, head_idx):
    """Identify parameter slices belonging to a specific attention head.

    In timm's ViT, the QKV weight is (3*embed_dim, embed_dim) where each
    head's Q/K/V occupies a contiguous slice of head_dim columns within
    the output dimension.

    Returns:
        list of (param_tensor, slice_or_index_info) for the head.
    """
    attn = model.blocks[block_idx].attn
    head_dim = attn.head_dim
    num_heads = attn.num_heads
    start = head_idx * head_dim
    end = start + head_dim

    head_param_info = []

    # QKV weight: shape (3*embed_dim, embed_dim)
    # Q rows: [start:end], K rows: [embed_dim+start:embed_dim+end], V rows: [2*embed_dim+start:2*embed_dim+end]
    embed_dim = num_heads * head_dim
    qkv_w = attn.qkv.weight
    q_rows = (qkv_w, slice(start, end))
    k_rows = (qkv_w, slice(embed_dim + start, embed_dim + end))
    v_rows = (qkv_w, slice(2 * embed_dim + start, 2 * embed_dim + end))
    head_param_info.extend([q_rows, k_rows, v_rows])

    # QKV bias if present
    if attn.qkv.bias is not None:
        qkv_b = attn.qkv.bias
        head_param_info.append((qkv_b, slice(start, end)))
        head_param_info.append((qkv_b, slice(embed_dim + start, embed_dim + end)))
        head_param_info.append((qkv_b, slice(2 * embed_dim + start, 2 * embed_dim + end)))

    # Output projection: weight shape (embed_dim, embed_dim)
    # Columns [start:end] correspond to head h's contribution
    proj_w = attn.proj.weight
    head_param_info.append((proj_w, (slice(None), slice(start, end))))

    return head_param_info


def count_head_params(model, block_idx, head_idx):
    """Count number of parameters belonging to a specific head."""
    info = get_head_params(model, block_idx, head_idx)
    total = 0
    for param, sl in info:
        total += param[sl].numel()
    return total


# ─── SGLD sampler ─────────────────────────────────────────────────────────────

def sgld_step(params_with_slices, grads_dict, lr, temperature):
    """One step of Stochastic Gradient Langevin Dynamics.

    θ_{t+1} = θ_t - lr * ∇L + sqrt(2 * lr / β) * η,  η ~ N(0, I)

    where β = 1/temperature.
    """
    with torch.no_grad():
        noise_scale = (2.0 * lr * temperature) ** 0.5
        for param, sl in params_with_slices:
            if param.grad is not None:
                g = param.grad[sl]
            else:
                g = torch.zeros_like(param[sl])
            noise = torch.randn_like(param[sl]) * noise_scale
            param[sl] -= lr * g - noise


@torch.no_grad()
def compute_loss_on_subset(model, loader, device, criterion, max_batches=None):
    """Compute average cross-entropy loss on a data subset."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for i, (images, targets) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images, targets = images.to(device), targets.to(device)
        out = model(images)
        loss = criterion(out, targets)
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
    return total_loss / max(1, total_samples)


def estimate_llc_for_head(model, block_idx, head_idx, loader, device, criterion,
                          sgld_steps=500, sgld_lr=1e-5, temperature=1.0,
                          max_batches=10):
    """Estimate LLC for a single head via SGLD.

    Returns:
        llc: estimated local learning coefficient.
        loss_star: loss at the converged point θ*.
        mean_sgld_loss: mean loss under SGLD posterior.
    """
    # Save original parameters
    head_params = get_head_params(model, block_idx, head_idx)
    original_values = [(p[sl].clone(),) for p, sl in head_params]

    # Loss at θ* (converged point)
    loss_star = compute_loss_on_subset(model, loader, device, criterion, max_batches)

    # Freeze all params except the head's
    for p in model.parameters():
        p.requires_grad_(False)
    for p, sl in head_params:
        p.requires_grad_(True)

    # SGLD chain
    sgld_losses = []
    data_iter = iter(loader)

    for step in range(sgld_steps):
        model.train()
        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, targets = next(data_iter)

        images, targets = images.to(device), targets.to(device)
        out = model(images)
        loss = criterion(out, targets)
        model.zero_grad()
        loss.backward()

        sgld_step(head_params, None, sgld_lr, temperature)
        sgld_losses.append(loss.item())

    mean_sgld_loss = float(np.mean(sgld_losses[len(sgld_losses) // 5:]))  # discard burn-in

    # Restore original parameters
    with torch.no_grad():
        for (p, sl), (orig,) in zip(head_params, original_values):
            p[sl].copy_(orig)

    # Re-enable all gradients
    for p in model.parameters():
        p.requires_grad_(True)

    # LLC ≈ n * (E_β[L] - L*) where n = dataset size
    n_samples = sum(images.size(0) for images, _ in loader)
    llc = n_samples * (mean_sgld_loss - loss_star)

    return llc, loss_star, mean_sgld_loss


# ─── Full analysis ────────────────────────────────────────────────────────────

def compute_all_llc(model, loader, device, cfg, sgld_steps, sgld_lr, num_chains):
    """Compute LLC for every head across all layers."""
    criterion = nn.CrossEntropyLoss()
    results = {}

    for b in range(cfg.num_blocks):
        results[b] = {}
        for h in range(cfg.num_heads):
            print(f"  Layer {b}, Head {h}...", end=" ", flush=True)
            llc_chains = []
            for chain in range(num_chains):
                llc, l_star, l_sgld = estimate_llc_for_head(
                    model, b, h, loader, device, criterion,
                    sgld_steps=sgld_steps, sgld_lr=sgld_lr,
                )
                llc_chains.append(llc)

            d_h = count_head_params(model, b, h)
            mean_llc = float(np.mean(llc_chains))
            results[b][h] = {
                "llc": mean_llc,
                "llc_normalised": mean_llc / (d_h / 2) if d_h > 0 else 0,
                "num_params": d_h,
                "chains": llc_chains,
            }
            print(f"LLC={mean_llc:.4f}, λ̂={results[b][h]['llc_normalised']:.4f}")

    return results


# ─── MAD for scatter plot ─────────────────────────────────────────────────────

@torch.no_grad()
def compute_all_mads(model, loader, device, cfg):
    """Compute per-head MAD averaged over eval subset."""
    model.eval()
    dist_matrix = build_distance_matrix(cfg.grid_h, cfg.grid_w, device=device)
    all_blocks = list(range(cfg.num_blocks))
    mad_accum = {b: [] for b in all_blocks}

    for images, _ in tqdm(loader, desc="Computing MADs", leave=False):
        images = images.to(device)
        with capture_attention(model, all_blocks) as get_attn:
            _ = model(images)
            attn_dict = get_attn()
        for b in all_blocks:
            mad_accum[b].append(compute_mad(attn_dict[b], dist_matrix).cpu().numpy())

    return {b: np.mean(np.stack(mad_accum[b], axis=0), axis=0) for b in all_blocks}


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_llc_heatmap(llc_results, save_path, key="llc_normalised",
                     title="Normalised LLC per Head"):
    layers = sorted(llc_results.keys())
    heads = sorted(llc_results[layers[0]].keys())
    mat = np.array([[llc_results[l][h][key] for h in heads] for l in layers])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        mat, ax=ax, cmap="coolwarm", annot=True, fmt=".3f",
        annot_kws={"fontsize": 7},
        xticklabels=[f"H{h}" for h in heads],
        yticklabels=[f"L{l}" for l in layers],
    )
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_llc_vs_mad(llc_results, mad_results, save_path):
    """Scatter plot: normalised LLC vs MAD, coloured by layer."""
    fig, ax = plt.subplots(figsize=(8, 6))
    layers = sorted(llc_results.keys())
    cmap = plt.cm.viridis(np.linspace(0, 1, len(layers)))

    for i, l in enumerate(layers):
        heads = sorted(llc_results[l].keys())
        llcs = [llc_results[l][h]["llc_normalised"] for h in heads]
        mads = [mad_results[l][h] for h in heads]
        ax.scatter(mads, llcs, color=cmap[i], label=f"L{l}", s=40, alpha=0.8,
                   edgecolors="black", linewidth=0.5)

    ax.set_xlabel("Mean Attention Distance (MAD)")
    ax.set_ylabel("Normalised LLC (λ̂)")
    ax.set_title("LLC vs MAD per Head")
    ax.legend(fontsize=6, ncol=3, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Per-head LLC (SLT complexity)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="analysis/llc")
    parser.add_argument("--sgld_steps", type=int, default=500)
    parser.add_argument("--sgld_lr", type=float, default=1e-5)
    parser.add_argument("--num_chains", type=int, default=3,
                        help="Number of independent SGLD chains for averaging.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = cfg.device
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    loader, _ = get_attention_eval_subset(cfg)

    print("Loading model...")
    model = load_vit_small(cfg, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # ── LLC ──
    print(f"Estimating LLC ({args.sgld_steps} SGLD steps × {args.num_chains} chains)...")
    llc_results = compute_all_llc(model, loader, device, cfg,
                                  args.sgld_steps, args.sgld_lr, args.num_chains)

    with open(os.path.join(out, "llc_per_head.json"), "w") as f:
        json.dump({str(b): {str(h): v for h, v in heads.items()}
                   for b, heads in llc_results.items()}, f, indent=2)

    plot_llc_heatmap(llc_results, os.path.join(out, "llc_heatmap.png"))
    plot_llc_heatmap(llc_results, os.path.join(out, "llc_raw_heatmap.png"),
                     key="llc", title="Raw LLC per Head")

    # ── MAD for correlation ──
    print("Computing MADs for LLC-vs-MAD scatter...")
    mad_results = compute_all_mads(model, loader, device, cfg)
    plot_llc_vs_mad(llc_results, mad_results, os.path.join(out, "llc_vs_mad.png"))

    # ── Summary statistics ──
    all_llc = []
    for b in sorted(llc_results.keys()):
        for h in sorted(llc_results[b].keys()):
            all_llc.append(llc_results[b][h]["llc_normalised"])
    print(f"\nLLC summary: mean={np.mean(all_llc):.4f}, std={np.std(all_llc):.4f}, "
          f"min={np.min(all_llc):.4f}, max={np.max(all_llc):.4f}")

    print("LLC analysis complete.")


if __name__ == "__main__":
    main()
