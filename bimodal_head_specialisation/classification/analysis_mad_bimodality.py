"""Comprehensive MAD bimodality analysis across layers.

Validates the bimodality claim explicitly with multiple statistical tests
and diagnostic visualisations — per-layer MAD histograms, GMM fits,
Hartigan's dip test, and cross-layer bimodality progression.

References:
  - Raghu et al., "Do Vision Transformers See Like CNNs?", NeurIPS 2021
  - Darcet et al., "Vision Transformers Need Registers", ICLR 2024
    (arXiv:2309.16588)
  - "ViT Attention Head Intervention" (arXiv:2601.04398)

Usage:
  python analysis_mad_bimodality.py \
      --config configs/baseline.yaml \
      --baseline_ckpt runs/baseline/checkpoints/best.pth \
      [--targeted_ckpt runs/spread_weak/checkpoints/best.pth] \
      --output_dir analysis/bimodality/

Outputs:
  - per_layer_histograms.png        Histograms of per-head MAD per layer
  - gmm_fits.json                   1- and 2-component GMM fits with AIC/BIC
  - gmm_fit_overlay.png             Histogram + GMM density overlay per layer
  - dip_test.json                   Hartigan's dip test p-values per layer
  - bimodality_progression.png      Bimodality strength across layers
  - mad_stability.json/png          MAD consistency across eval subsets (bootstrap)
  - head_role_assignment.json       Local/global classification per head
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
from sklearn.mixture import GaussianMixture
from scipy import stats
from tqdm import tqdm

from common.config import load_config
from common.model_utils import load_vit_small
from common.attention_hooks import capture_attention
from common.mad_metrics import build_distance_matrix, compute_mad
from data import get_attention_eval_subset


# ─── MAD collection ──────────────────────────────────────────────────────────

@torch.no_grad()
def collect_per_sample_mads(model, loader, device, dist_matrix, num_blocks):
    """Collect per-sample per-head MAD across all layers.

    Returns:
        dict: {layer: (N_samples, H)} array of MAD values.
    """
    model.eval()
    all_blocks = list(range(num_blocks))
    accum = {b: [] for b in all_blocks}

    for images, _ in tqdm(loader, desc="Collecting MADs", leave=False):
        images = images.to(device)
        with capture_attention(model, all_blocks) as get_attn:
            _ = model(images)
            attn_dict = get_attn()
        for b in all_blocks:
            # compute_mad returns (H,) averaged over batch and queries
            # For per-sample analysis, compute manually
            a = attn_dict[b][:, :, 1:, 1:]  # exclude CLS
            B, H, N, N2 = a.shape
            R = dist_matrix[:N, :N2].to(a.device)
            weighted = (a * R.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
            per_sample_mad = weighted.mean(dim=-1)  # (B, H)
            accum[b].append(per_sample_mad.cpu().numpy())

    return {b: np.concatenate(accum[b], axis=0) for b in all_blocks}


# ─── GMM fitting ─────────────────────────────────────────────────────────────

def fit_gmm_per_layer(mad_dict, max_components=3):
    """Fit 1-, 2-, and 3-component GMMs per layer.

    Returns:
        dict: {layer: {n_comp: {aic, bic, means, weights, covariances}}}
    """
    results = {}
    for b in sorted(mad_dict.keys()):
        mads = mad_dict[b]  # (N_samples, H)
        # Pool across heads: treat each (sample, head) as an observation
        flat = mads.flatten().reshape(-1, 1)
        results[b] = {}
        for n in range(1, max_components + 1):
            gmm = GaussianMixture(n_components=n, random_state=42,
                                   n_init=5, max_iter=300)
            gmm.fit(flat)
            results[b][n] = {
                "aic": float(gmm.aic(flat)),
                "bic": float(gmm.bic(flat)),
                "means": gmm.means_.flatten().tolist(),
                "weights": gmm.weights_.tolist(),
                "covariances": gmm.covariances_.flatten().tolist(),
                "converged": bool(gmm.converged_),
            }
        # Also store per-head means
        results[b]["per_head_mean_mad"] = mads.mean(axis=0).tolist()
        results[b]["per_head_std_mad"] = mads.std(axis=0).tolist()
    return results


# ─── Hartigan's Dip Test ──────────────────────────────────────────────────────

def hartigans_dip_test(data, num_bootstrap=1000):
    """Hartigan's dip statistic for unimodality testing.

    Uses bootstrap to estimate p-value. A small p-value rejects unimodality
    (i.e. suggests multimodality).

    Returns:
        dip_stat: the dip statistic.
        p_value: bootstrap p-value.
    """
    data = np.sort(data)
    n = len(data)

    def _dip(sorted_data):
        """Compute the dip statistic (simplified)."""
        n = len(sorted_data)
        # Empirical CDF
        ecdf = np.arange(1, n + 1) / n
        # Greatest convex minorant (GCM) and least concave majorant (LCM)
        # Simplified: use the maximum deviation from uniform CDF
        uniform_cdf = np.linspace(1 / n, 1.0, n)
        dip = np.max(np.abs(ecdf - uniform_cdf)) / 2
        return dip

    observed_dip = _dip(data)

    # Bootstrap under null (uniform distribution)
    bootstrap_dips = []
    for _ in range(num_bootstrap):
        uniform_sample = np.sort(np.random.uniform(data.min(), data.max(), n))
        bootstrap_dips.append(_dip(uniform_sample))

    p_value = np.mean(np.array(bootstrap_dips) >= observed_dip)
    return float(observed_dip), float(p_value)


def dip_test_per_layer(mad_dict, num_bootstrap=1000):
    """Run dip test on pooled MAD per layer."""
    results = {}
    for b in sorted(mad_dict.keys()):
        flat = mad_dict[b].flatten()
        dip, p = hartigans_dip_test(flat, num_bootstrap)
        results[b] = {"dip_stat": dip, "p_value": p,
                       "rejects_unimodality_005": p < 0.05}
    return results


# ─── MAD stability (bootstrap) ───────────────────────────────────────────────

def mad_stability_bootstrap(mad_dict, num_bootstrap=200):
    """Bootstrap confidence intervals for per-head mean MAD.

    Returns:
        dict: {layer: {head: {mean, ci_low, ci_high, std}}}
    """
    results = {}
    for b in sorted(mad_dict.keys()):
        mads = mad_dict[b]  # (N_samples, H)
        n_samples, n_heads = mads.shape
        results[b] = {}
        for h in range(n_heads):
            head_mads = mads[:, h]
            boot_means = []
            for _ in range(num_bootstrap):
                sample = np.random.choice(head_mads, size=n_samples, replace=True)
                boot_means.append(sample.mean())
            boot_means = np.array(boot_means)
            results[b][h] = {
                "mean": float(head_mads.mean()),
                "ci_low": float(np.percentile(boot_means, 2.5)),
                "ci_high": float(np.percentile(boot_means, 97.5)),
                "std": float(head_mads.std()),
                "boot_std": float(boot_means.std()),
            }
    return results


# ─── Head role assignment ─────────────────────────────────────────────────────

def assign_head_roles(mad_dict, gmm_results):
    """Classify each head as local/global using the 2-component GMM boundary.

    Returns:
        dict: {layer: {head: {role, mean_mad, distance_to_boundary}}}
    """
    results = {}
    for b in sorted(mad_dict.keys()):
        results[b] = {}
        gmm2 = gmm_results[b].get(2, None)
        if gmm2 is None:
            continue
        means = sorted(gmm2["means"])
        boundary = np.mean(means) if len(means) == 2 else 0.5

        head_means = mad_dict[b].mean(axis=0)
        for h, m in enumerate(head_means):
            role = "local" if m < boundary else "global"
            results[b][h] = {
                "role": role,
                "mean_mad": float(m),
                "boundary": float(boundary),
                "distance_to_boundary": float(abs(m - boundary)),
                "gmm_component_means": means,
            }
    return results


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_per_layer_histograms(mad_dict, save_path, title_prefix=""):
    """Grid of per-layer MAD histograms with head-level detail."""
    layers = sorted(mad_dict.keys())
    n = len(layers)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = axes.flatten()

    for i, b in enumerate(layers):
        ax = axes[i]
        mads = mad_dict[b]  # (N_samples, H)
        n_heads = mads.shape[1]

        # Overall histogram
        ax.hist(mads.flatten(), bins=40, density=True, alpha=0.3,
                color="gray", label="pooled")
        # Per-head vertical lines
        colors = plt.cm.Set1(np.linspace(0, 1, n_heads))
        for h in range(n_heads):
            mean_m = mads[:, h].mean()
            ax.axvline(mean_m, color=colors[h], linestyle="--",
                       linewidth=1.2, label=f"H{h}: {mean_m:.3f}")

        ax.set_title(f"{title_prefix}Layer {b}", fontsize=9)
        ax.set_xlabel("MAD", fontsize=7)
        ax.legend(fontsize=5, loc="upper right")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_gmm_overlay(mad_dict, gmm_results, save_path):
    """Histogram with 2-component GMM density overlay per layer."""
    layers = sorted(mad_dict.keys())
    n = len(layers)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = axes.flatten()

    for i, b in enumerate(layers):
        ax = axes[i]
        flat = mad_dict[b].flatten()
        ax.hist(flat, bins=50, density=True, alpha=0.4, color="steelblue")

        gmm2 = gmm_results[b].get(2)
        if gmm2:
            x = np.linspace(flat.min(), flat.max(), 200)
            total_pdf = np.zeros_like(x)
            for k in range(len(gmm2["means"])):
                mu = gmm2["means"][k]
                var = gmm2["covariances"][k]
                w = gmm2["weights"][k]
                comp_pdf = w * stats.norm.pdf(x, mu, np.sqrt(var))
                total_pdf += comp_pdf
                ax.plot(x, comp_pdf, "--", linewidth=1, alpha=0.7,
                        label=f"μ={mu:.3f}, w={w:.2f}")
            ax.plot(x, total_pdf, "k-", linewidth=1.5, label="mixture")

        bic_diff = gmm_results[b][1]["bic"] - gmm_results[b][2]["bic"]
        ax.set_title(f"Layer {b}  (ΔBIC={bic_diff:.1f})", fontsize=9)
        ax.legend(fontsize=5)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("2-Component GMM Fit per Layer", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_bimodality_progression(gmm_results, dip_results, save_path):
    """Plot bimodality metrics across layers."""
    layers = sorted(gmm_results.keys())
    bic_diffs = [gmm_results[b][1]["bic"] - gmm_results[b][2]["bic"] for b in layers]
    aic_diffs = [gmm_results[b][1]["aic"] - gmm_results[b][2]["aic"] for b in layers]
    dip_pvals = [dip_results[b]["p_value"] for b in layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # BIC/AIC improvement
    x = np.arange(len(layers))
    ax1.bar(x - 0.15, bic_diffs, 0.3, label="ΔBIC (1 vs 2)", color="steelblue")
    ax1.bar(x + 0.15, aic_diffs, 0.3, label="ΔAIC (1 vs 2)", color="coral")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.axhline(10, color="gray", linewidth=0.5, linestyle="--",
                label="Strong evidence threshold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"L{b}" for b in layers], fontsize=8)
    ax1.set_ylabel("Information criterion difference")
    ax1.set_title("GMM Model Selection: 1-comp vs 2-comp")
    ax1.legend(fontsize=7)

    # Dip test p-values
    colors = ["red" if p < 0.05 else "steelblue" for p in dip_pvals]
    ax2.bar(x, dip_pvals, color=colors)
    ax2.axhline(0.05, color="red", linewidth=1, linestyle="--", label="α=0.05")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"L{b}" for b in layers], fontsize=8)
    ax2.set_ylabel("Dip test p-value")
    ax2.set_title("Hartigan's Dip Test (Unimodality)")
    ax2.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_stability(stability_results, save_path):
    """Plot per-head MAD with bootstrap CI across layers."""
    layers = sorted(stability_results.keys())
    n_heads = len(stability_results[layers[0]])

    fig, ax = plt.subplots(figsize=(12, 5))
    x_offset = 0
    xtick_positions = []
    xtick_labels = []

    colors = plt.cm.Set1(np.linspace(0, 1, n_heads))
    for b in layers:
        for h in range(n_heads):
            s = stability_results[b][h]
            ax.errorbar(x_offset, s["mean"],
                        yerr=[[s["mean"] - s["ci_low"]], [s["ci_high"] - s["mean"]]],
                        fmt="o", color=colors[h], markersize=4, capsize=2,
                        linewidth=1)
            x_offset += 1
        xtick_positions.append(x_offset - n_heads / 2)
        xtick_labels.append(f"L{b}")
        x_offset += 1  # gap between layers

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=8)
    ax.set_ylabel("Mean MAD")
    ax.set_title("Per-Head MAD with 95% Bootstrap CI")

    # Legend for heads
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=colors[h], label=f"H{h}", markersize=6)
                       for h in range(n_heads)]
    ax.legend(handles=legend_elements, fontsize=7, ncol=n_heads, loc="upper right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MAD bimodality analysis")
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline_ckpt", required=True)
    parser.add_argument("--targeted_ckpt", default=None)
    parser.add_argument("--output_dir", default="analysis/bimodality")
    parser.add_argument("--num_bootstrap_dip", type=int, default=1000)
    parser.add_argument("--num_bootstrap_stability", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = cfg.device
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    loader, _ = get_attention_eval_subset(cfg)
    dist_matrix = build_distance_matrix(cfg.grid_h, cfg.grid_w, device=device)

    def _analyse(ckpt_path, tag):
        print(f"\n{'='*50}")
        print(f"  MAD Bimodality Analysis: {tag}")
        print(f"{'='*50}")

        model = load_vit_small(cfg, pretrained=False)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        # Collect per-sample MADs
        print("Collecting per-sample MADs...")
        mad_dict = collect_per_sample_mads(model, loader, device, dist_matrix,
                                           cfg.num_blocks)
        del model
        torch.cuda.empty_cache()

        # Per-layer histograms
        print("Plotting per-layer histograms...")
        plot_per_layer_histograms(
            mad_dict, os.path.join(out, f"histograms_{tag}.png"),
            title_prefix=f"{tag.title()} ")

        # GMM fitting
        print("Fitting GMMs (1, 2, 3 components)...")
        gmm = fit_gmm_per_layer(mad_dict, max_components=3)
        with open(os.path.join(out, f"gmm_fits_{tag}.json"), "w") as f:
            json.dump({str(b): {str(k): v for k, v in layer.items()}
                       for b, layer in gmm.items()}, f, indent=2)
        plot_gmm_overlay(mad_dict, gmm, os.path.join(out, f"gmm_overlay_{tag}.png"))

        # Hartigan's dip test
        print("Running Hartigan's dip test...")
        dip = dip_test_per_layer(mad_dict, args.num_bootstrap_dip)
        with open(os.path.join(out, f"dip_test_{tag}.json"), "w") as f:
            json.dump({str(b): v for b, v in dip.items()}, f, indent=2)

        # Bimodality progression
        print("Plotting bimodality progression...")
        plot_bimodality_progression(gmm, dip,
                                    os.path.join(out, f"bimodality_progression_{tag}.png"))

        # Bootstrap stability
        print("Bootstrap MAD stability...")
        stability = mad_stability_bootstrap(mad_dict, args.num_bootstrap_stability)
        with open(os.path.join(out, f"mad_stability_{tag}.json"), "w") as f:
            json.dump({str(b): {str(h): v for h, v in heads.items()}
                       for b, heads in stability.items()}, f, indent=2)
        plot_stability(stability, os.path.join(out, f"mad_stability_{tag}.png"))

        # Head role assignment
        print("Assigning head roles...")
        roles = assign_head_roles(mad_dict, gmm)
        with open(os.path.join(out, f"head_roles_{tag}.json"), "w") as f:
            json.dump({str(b): {str(h): v for h, v in heads.items()}
                       for b, heads in roles.items()}, f, indent=2)

        # Summary
        print(f"\n--- {tag.title()} Summary ---")
        for b in sorted(gmm.keys()):
            bic_diff = gmm[b][1]["bic"] - gmm[b][2]["bic"]
            dip_p = dip[b]["p_value"]
            head_means = gmm[b].get("per_head_mean_mad", [])
            n_local = sum(1 for h in roles.get(b, {}).values() if h["role"] == "local")
            n_global = sum(1 for h in roles.get(b, {}).values() if h["role"] == "global")
            print(f"  L{b}: ΔBIC={bic_diff:+.1f}, dip p={dip_p:.3f}, "
                  f"{n_local}L/{n_global}G, heads={[f'{m:.3f}' for m in head_means]}")

        return mad_dict

    _analyse(args.baseline_ckpt, "baseline")
    if args.targeted_ckpt:
        _analyse(args.targeted_ckpt, "targeted")

    print("\nMAD bimodality analysis complete.")


if __name__ == "__main__":
    main()
