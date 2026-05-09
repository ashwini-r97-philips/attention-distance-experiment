"""CKA (Centered Kernel Alignment) representation similarity analysis.

Compares layer-wise representations between baseline and regularised models
(or cross-layer within a single model).

Reference: Kornblith et al., "Similarity of Neural Network Representations
Revisited", ICML 2019 (arXiv:1905.00414).

Usage:
  python analysis_cka.py \
      --config configs/baseline.yaml \
      --baseline_ckpt runs/baseline/checkpoints/best.pth \
      --targeted_ckpt runs/spread_weak/checkpoints/best.pth \
      --output_dir analysis/cka/

Outputs:
  - cka_self_baseline.json/png     Self-CKA matrix for baseline (layer × layer)
  - cka_self_targeted.json/png     Self-CKA matrix for targeted
  - cka_cross.json/png             Cross-model CKA (baseline layer × targeted layer)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from common.config import load_config
from common.model_utils import load_vit_small
from data import get_attention_eval_subset


# ─── CKA core ────────────────────────────────────────────────────────────────

def _center_gram(G):
    """Center a Gram matrix: H G H where H = I - (1/n)11^T."""
    n = G.shape[0]
    ones = torch.ones(n, n, device=G.device, dtype=G.dtype) / n
    return G - ones @ G - G @ ones + ones @ G @ ones


def linear_cka(X, Y):
    """Linear CKA between two representation matrices.

    Args:
        X: (n, d1) representations from model/layer 1.
        Y: (n, d2) representations from model/layer 2.

    Returns:
        Scalar CKA similarity in [0, 1].
    """
    Gx = X @ X.T
    Gy = Y @ Y.T
    Gx = _center_gram(Gx)
    Gy = _center_gram(Gy)
    hsic_xy = (Gx * Gy).sum()
    hsic_xx = (Gx * Gx).sum()
    hsic_yy = (Gy * Gy).sum()
    denom = (hsic_xx * hsic_yy).sqrt().clamp(min=1e-12)
    return (hsic_xy / denom).item()


# ─── Feature extraction ──────────────────────────────────────────────────────

def _register_hooks(model):
    """Register forward hooks on every block to capture CLS + patch token output."""
    features = {}
    handles = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output: (B, N, D) — store detached
            features[layer_idx] = output.detach()
        return hook

    for idx, block in enumerate(model.blocks):
        h = block.register_forward_hook(make_hook(idx))
        handles.append(h)
    return features, handles


@torch.no_grad()
def extract_representations(model, loader, device, num_blocks):
    """Extract per-layer representations over the eval subset.

    Returns:
        dict: {layer_idx: (N_total, D)} where D = embed_dim, tokens averaged
              over spatial dimension (global average pool over patch tokens).
    """
    model.eval()
    features_ref, handles = _register_hooks(model)
    accum = {b: [] for b in range(num_blocks)}

    for images, _ in tqdm(loader, desc="Extracting features", leave=False):
        images = images.to(device)
        _ = model(images)
        for b in range(num_blocks):
            # Average pool patch tokens (skip CLS at position 0)
            patch_tokens = features_ref[b][:, 1:, :]  # (B, N_patches, D)
            pooled = patch_tokens.mean(dim=1)           # (B, D)
            accum[b].append(pooled.cpu())

    for h in handles:
        h.remove()

    return {b: torch.cat(accum[b], dim=0) for b in range(num_blocks)}


# ─── CKA matrix computation ──────────────────────────────────────────────────

def compute_cka_matrix(reps_a, reps_b, device="cpu"):
    """Compute CKA between all layer pairs.

    Args:
        reps_a: {layer: (N, D)} from model A.
        reps_b: {layer: (N, D)} from model B.

    Returns:
        np.array of shape (num_layers_a, num_layers_b).
    """
    layers_a = sorted(reps_a.keys())
    layers_b = sorted(reps_b.keys())
    mat = np.zeros((len(layers_a), len(layers_b)))
    for i, la in enumerate(tqdm(layers_a, desc="CKA rows", leave=False)):
        X = reps_a[la].to(device).float()
        for j, lb in enumerate(layers_b):
            Y = reps_b[lb].to(device).float()
            mat[i, j] = linear_cka(X, Y)
    return mat


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_cka_matrix(mat, save_path, title="CKA Similarity", labels_x=None, labels_y=None):
    fig, ax = plt.subplots(figsize=(8, 7))
    n_y, n_x = mat.shape
    if labels_x is None:
        labels_x = [str(i) for i in range(n_x)]
    if labels_y is None:
        labels_y = [str(i) for i in range(n_y)]
    sns.heatmap(
        mat, ax=ax, vmin=0, vmax=1, cmap="magma",
        xticklabels=labels_x, yticklabels=labels_y,
        annot=True, fmt=".2f", annot_kws={"fontsize": 7},
        square=True,
    )
    ax.set_xlabel("Layer (Model B)" if "Cross" in title else "Layer")
    ax.set_ylabel("Layer (Model A)" if "Cross" in title else "Layer")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CKA representation similarity")
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline_ckpt", required=True)
    parser.add_argument("--targeted_ckpt", default=None,
                        help="If omitted, only self-CKA for baseline is computed.")
    parser.add_argument("--output_dir", default="analysis/cka")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = cfg.device
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    # Load eval subset
    loader, _ = get_attention_eval_subset(cfg)

    # ── Baseline features ──
    print("Loading baseline model...")
    model_b = load_vit_small(cfg, pretrained=False)
    ckpt = torch.load(args.baseline_ckpt, map_location=device, weights_only=False)
    model_b.load_state_dict(ckpt["model_state_dict"])

    print("Extracting baseline representations...")
    reps_baseline = extract_representations(model_b, loader, device, cfg.num_blocks)
    del model_b
    torch.cuda.empty_cache()

    # ── Self-CKA baseline ──
    print("Computing self-CKA (baseline)...")
    self_cka_b = compute_cka_matrix(reps_baseline, reps_baseline, device=device)
    with open(os.path.join(out, "cka_self_baseline.json"), "w") as f:
        json.dump({"matrix": self_cka_b.tolist()}, f, indent=2)
    plot_cka_matrix(self_cka_b, os.path.join(out, "cka_self_baseline.png"),
                    title="Self-CKA (Baseline)")

    if args.targeted_ckpt:
        # ── Targeted features ──
        print("Loading targeted model...")
        model_t = load_vit_small(cfg, pretrained=False)
        ckpt_t = torch.load(args.targeted_ckpt, map_location=device, weights_only=False)
        model_t.load_state_dict(ckpt_t["model_state_dict"])

        print("Extracting targeted representations...")
        reps_targeted = extract_representations(model_t, loader, device, cfg.num_blocks)
        del model_t
        torch.cuda.empty_cache()

        # ── Self-CKA targeted ──
        print("Computing self-CKA (targeted)...")
        self_cka_t = compute_cka_matrix(reps_targeted, reps_targeted, device=device)
        with open(os.path.join(out, "cka_self_targeted.json"), "w") as f:
            json.dump({"matrix": self_cka_t.tolist()}, f, indent=2)
        plot_cka_matrix(self_cka_t, os.path.join(out, "cka_self_targeted.png"),
                        title="Self-CKA (Targeted)")

        # ── Cross-model CKA ──
        print("Computing cross-model CKA...")
        cross_cka = compute_cka_matrix(reps_baseline, reps_targeted, device=device)
        with open(os.path.join(out, "cka_cross.json"), "w") as f:
            json.dump({"matrix": cross_cka.tolist()}, f, indent=2)
        plot_cka_matrix(cross_cka, os.path.join(out, "cka_cross.png"),
                        title="Cross-Model CKA (Baseline × Targeted)",
                        labels_x=[f"T{i}" for i in range(cfg.num_blocks)],
                        labels_y=[f"B{i}" for i in range(cfg.num_blocks)])

        # ── Diagonal summary (same-layer agreement) ──
        diag = np.diag(cross_cka).tolist()
        print(f"Cross-model CKA diagonal: {[round(v, 3) for v in diag]}")

    print("CKA analysis complete.")


if __name__ == "__main__":
    main()
