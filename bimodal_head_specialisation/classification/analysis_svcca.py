"""SVCCA (Singular Vector Canonical Correlation Analysis) subspace alignment.

Measures how aligned the representation subspaces are between baseline and
regularised models, or across layers within a single model.

Reference: Raghu et al., "SVCCA: Singular Vector Canonical Correlation for
Deep Learning Dynamics and Interpretability", NeurIPS 2017.

Usage:
  python analysis_svcca.py \
      --config configs/baseline.yaml \
      --baseline_ckpt runs/baseline/checkpoints/best.pth \
      --targeted_ckpt runs/spread_weak/checkpoints/best.pth \
      --output_dir analysis/svcca/

Outputs:
  - svcca_self_baseline.json/png   Self-SVCCA (layer × layer)
  - svcca_cross.json/png           Cross-model SVCCA
  - svcca_spectrum.png             Singular value spectrum per layer
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


# ─── Feature extraction (same pattern as CKA) ────────────────────────────────

def _register_hooks(model):
    features = {}
    handles = []

    def make_hook(idx):
        def hook(module, input, output):
            features[idx] = output.detach()
        return hook

    for idx, block in enumerate(model.blocks):
        h = block.register_forward_hook(make_hook(idx))
        handles.append(h)
    return features, handles


@torch.no_grad()
def extract_representations(model, loader, device, num_blocks):
    """Extract per-layer patch-averaged representations.

    Returns:
        dict: {layer: (N_samples, D)}.
    """
    model.eval()
    features_ref, handles = _register_hooks(model)
    accum = {b: [] for b in range(num_blocks)}

    for images, _ in tqdm(loader, desc="Extracting features", leave=False):
        images = images.to(device)
        _ = model(images)
        for b in range(num_blocks):
            patch_tokens = features_ref[b][:, 1:, :]  # skip CLS
            pooled = patch_tokens.mean(dim=1)
            accum[b].append(pooled.cpu())

    for h in handles:
        h.remove()
    return {b: torch.cat(accum[b], dim=0).numpy() for b in range(num_blocks)}


# ─── SVCCA core ──────────────────────────────────────────────────────────────

def _svd_reduce(X, variance_threshold=0.99):
    """Reduce X to top-k SVD components that explain `variance_threshold` variance.

    Args:
        X: (n, d) centered representation matrix.
        variance_threshold: fraction of variance to retain.

    Returns:
        X_reduced: (n, k) projected onto top-k singular vectors.
        k: number of components retained.
    """
    X = X - X.mean(axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    var_explained = np.cumsum(s ** 2) / np.sum(s ** 2)
    k = np.searchsorted(var_explained, variance_threshold) + 1
    k = max(k, 1)
    return U[:, :k] * s[:k], k, s


def svcca_similarity(X, Y, variance_threshold=0.99):
    """Compute SVCCA similarity between two representation matrices.

    Steps:
        1. SVD-reduce both X and Y (keep components explaining 99% variance).
        2. Run CCA on the reduced representations.
        3. Return mean of canonical correlations.

    Args:
        X: (n, d1) representations from model/layer A.
        Y: (n, d2) representations from model/layer B.
        variance_threshold: variance explained threshold for SVD truncation.

    Returns:
        mean_cc: mean canonical correlation (scalar in [0, 1]).
        ccs: array of canonical correlations.
        k_x, k_y: number of SVD components retained.
    """
    X_r, k_x, _ = _svd_reduce(X, variance_threshold)
    Y_r, k_y, _ = _svd_reduce(Y, variance_threshold)

    # CCA: find linear projections that maximise correlation
    # Use QR decomposition for numerical stability
    Q_x, _ = np.linalg.qr(X_r)
    Q_y, _ = np.linalg.qr(Y_r)

    k = min(Q_x.shape[1], Q_y.shape[1])
    Q_x = Q_x[:, :k]
    Q_y = Q_y[:, :k]

    M = Q_x.T @ Q_y  # (k, k)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    # Canonical correlations are the singular values (clamped to [0,1])
    ccs = np.clip(s, 0.0, 1.0)

    return float(np.mean(ccs)), ccs, k_x, k_y


def compute_svcca_matrix(reps_a, reps_b, variance_threshold=0.99):
    """Compute SVCCA between all layer pairs.

    Returns:
        mat: (num_layers_a, num_layers_b) mean canonical correlations.
    """
    layers_a = sorted(reps_a.keys())
    layers_b = sorted(reps_b.keys())
    mat = np.zeros((len(layers_a), len(layers_b)))

    for i, la in enumerate(tqdm(layers_a, desc="SVCCA rows", leave=False)):
        for j, lb in enumerate(layers_b):
            cc, _, _, _ = svcca_similarity(reps_a[la], reps_b[lb], variance_threshold)
            mat[i, j] = cc
    return mat


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_svcca_matrix(mat, save_path, title="SVCCA Similarity",
                      labels_x=None, labels_y=None):
    fig, ax = plt.subplots(figsize=(8, 7))
    n_y, n_x = mat.shape
    if labels_x is None:
        labels_x = [str(i) for i in range(n_x)]
    if labels_y is None:
        labels_y = [str(i) for i in range(n_y)]
    sns.heatmap(
        mat, ax=ax, vmin=0, vmax=1, cmap="viridis",
        xticklabels=labels_x, yticklabels=labels_y,
        annot=True, fmt=".2f", annot_kws={"fontsize": 7},
        square=True,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_sv_spectrum(reps, save_path, title="Singular Value Spectrum"):
    """Plot the singular value spectrum for each layer."""
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = sorted(reps.keys())
    for l in layers:
        X = reps[l] - reps[l].mean(axis=0, keepdims=True)
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        s_norm = s / s.sum()
        ax.plot(s_norm[:50], label=f"L{l}", alpha=0.7, linewidth=1.2)

    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Normalized singular value")
    ax.set_title(title)
    ax.legend(fontsize=6, ncol=3, loc="upper right")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SVCCA subspace alignment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline_ckpt", required=True)
    parser.add_argument("--targeted_ckpt", default=None)
    parser.add_argument("--output_dir", default="analysis/svcca")
    parser.add_argument("--variance_threshold", type=float, default=0.99,
                        help="Fraction of variance retained in SVD step.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = cfg.device
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    loader, _ = get_attention_eval_subset(cfg)

    # ── Baseline ──
    print("Loading baseline model...")
    model_b = load_vit_small(cfg, pretrained=False)
    ckpt = torch.load(args.baseline_ckpt, map_location=device, weights_only=False)
    model_b.load_state_dict(ckpt["model_state_dict"])

    print("Extracting baseline representations...")
    reps_baseline = extract_representations(model_b, loader, device, cfg.num_blocks)
    del model_b
    torch.cuda.empty_cache()

    # Self-SVCCA (baseline)
    print("Computing self-SVCCA (baseline)...")
    self_svcca = compute_svcca_matrix(reps_baseline, reps_baseline, args.variance_threshold)
    with open(os.path.join(out, "svcca_self_baseline.json"), "w") as f:
        json.dump({"matrix": self_svcca.tolist()}, f, indent=2)
    plot_svcca_matrix(self_svcca, os.path.join(out, "svcca_self_baseline.png"),
                      title="Self-SVCCA (Baseline)")

    # SV spectrum
    plot_sv_spectrum(reps_baseline, os.path.join(out, "sv_spectrum_baseline.png"),
                     title="SV Spectrum (Baseline)")

    if args.targeted_ckpt:
        print("Loading targeted model...")
        model_t = load_vit_small(cfg, pretrained=False)
        ckpt_t = torch.load(args.targeted_ckpt, map_location=device, weights_only=False)
        model_t.load_state_dict(ckpt_t["model_state_dict"])

        print("Extracting targeted representations...")
        reps_targeted = extract_representations(model_t, loader, device, cfg.num_blocks)
        del model_t
        torch.cuda.empty_cache()

        # Self-SVCCA (targeted)
        print("Computing self-SVCCA (targeted)...")
        self_svcca_t = compute_svcca_matrix(reps_targeted, reps_targeted, args.variance_threshold)
        with open(os.path.join(out, "svcca_self_targeted.json"), "w") as f:
            json.dump({"matrix": self_svcca_t.tolist()}, f, indent=2)
        plot_svcca_matrix(self_svcca_t, os.path.join(out, "svcca_self_targeted.png"),
                          title="Self-SVCCA (Targeted)")

        plot_sv_spectrum(reps_targeted, os.path.join(out, "sv_spectrum_targeted.png"),
                         title="SV Spectrum (Targeted)")

        # Cross-model SVCCA
        print("Computing cross-model SVCCA...")
        cross = compute_svcca_matrix(reps_baseline, reps_targeted, args.variance_threshold)
        with open(os.path.join(out, "svcca_cross.json"), "w") as f:
            json.dump({"matrix": cross.tolist()}, f, indent=2)
        plot_svcca_matrix(cross, os.path.join(out, "svcca_cross.png"),
                          title="Cross-Model SVCCA (Baseline × Targeted)",
                          labels_x=[f"T{i}" for i in range(cfg.num_blocks)],
                          labels_y=[f"B{i}" for i in range(cfg.num_blocks)])

        diag = np.diag(cross).tolist()
        print(f"Cross SVCCA diagonal: {[round(v, 3) for v in diag]}")

    print("SVCCA analysis complete.")


if __name__ == "__main__":
    main()
