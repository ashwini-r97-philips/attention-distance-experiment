"""Head variance & head ordering analyses across all 3 runs.

A. Layer-wise Head Variance: Var(MAD across heads) per layer, all 3 runs overlaid.
C. Head Ordering: Heads sorted by MAD per layer, all 3 runs compared.

Reads only from existing evaluation_results.json files — no GPU needed.

Usage:
  python analysis_head_variance_ordering.py \
      --output_dir visualizations/head_analysis
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─── Config ──────────────────────────────────────────────────────────────────

RUNS = {
    "baseline": "runs/baseline_vit_s16_imagenet1k",
    "bimodal":  "runs/bimodal_weak_vit_s16_imagenet1k",
    "spread":   "runs/spread_weak_vit_s16_imagenet1k",
}

COLORS = {
    "baseline": "#4C72B0",
    "bimodal":  "#DD8452",
    "spread":   "#55A868",
}

MARKERS = {
    "baseline": "o",
    "bimodal":  "s",
    "spread":   "D",
}


def load_eval(run_dir):
    path = os.path.join(run_dir, "eval", "evaluation_results.json")
    with open(path) as f:
        return json.load(f)


def extract_mad_matrix(eval_data):
    """Return (num_layers, num_heads) array of MAD values."""
    attn = eval_data["attention_stats"]
    layers = sorted(attn.keys(), key=lambda x: int(x))
    mat = np.array([attn[l]["mad"] for l in layers])
    return mat, [int(l) for l in layers]


def load_epoch_mads(run_dir):
    """Load per-epoch MAD from attention_stats/epoch_XXXX.json."""
    stats_dir = os.path.join(run_dir, "attention_stats")
    if not os.path.isdir(stats_dir):
        return None
    epoch_files = sorted([f for f in os.listdir(stats_dir) if f.endswith(".json")])
    epoch_data = []
    for ef in epoch_files:
        with open(os.path.join(stats_dir, ef)) as f:
            d = json.load(f)
        layers = sorted(d.keys(), key=lambda x: int(x))
        mat = np.array([d[l]["mad"] for l in layers])
        epoch_data.append(mat)
    return np.stack(epoch_data, axis=0)  # (E, L, H)


# ═══════════════════════════════════════════════════════════════════════════════
# A. LAYER-WISE HEAD VARIANCE PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_layerwise_variance(all_mads, layers, out_dir):
    """Var(MAD across heads) per layer, all runs overlaid.
    
    Main figure: bar chart with lines.
    Sub-figure: separated into early/mid/late layer groups.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── Figure 1: Full variance profile ──
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(layers))
    width = 0.25
    offsets = {"baseline": -width, "bimodal": 0, "spread": width}

    for name, mad_mat in all_mads.items():
        var_per_layer = np.var(mad_mat, axis=1)
        ax.bar(x + offsets[name], var_per_layer, width, label=name.capitalize(),
               color=COLORS[name], alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Var(MAD across heads)", fontsize=12)
    ax.set_title("Inter-Head MAD Variance by Layer", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "layerwise_head_variance.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ── Figure 2: Line plot with confidence regions from epoch data ──
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, mad_mat in all_mads.items():
        var_per_layer = np.var(mad_mat, axis=1)
        ax.plot(layers, var_per_layer, marker=MARKERS[name], label=name.capitalize(),
                color=COLORS[name], linewidth=2, markersize=7)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Var(MAD across heads)", fontsize=12)
    ax.set_title("Head MAD Variance Profile Across Layers", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{l}" for l in layers])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "layerwise_head_variance_line.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ── Figure 3: Early / Mid / Late decomposition ──
    n = len(layers)
    third = n // 3
    groups = [
        ("Early (L0–L3)", slice(0, third)),
        ("Mid (L4–L7)", slice(third, 2 * third)),
        ("Late (L8–L11)", slice(2 * third, n)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (title, sl) in zip(axes, groups):
        sub_layers = layers[sl]
        sub_x = np.arange(len(sub_layers))
        for name, mad_mat in all_mads.items():
            var_per_layer = np.var(mad_mat[sl], axis=1)
            ax.bar(sub_x + offsets[name], var_per_layer, width,
                   label=name.capitalize(), color=COLORS[name], alpha=0.85,
                   edgecolor="white", linewidth=0.5)
        ax.set_xticks(sub_x)
        ax.set_xticklabels([f"L{l}" for l in sub_layers])
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        if ax == axes[0]:
            ax.set_ylabel("Var(MAD)", fontsize=11)
            ax.legend(fontsize=9)
    fig.suptitle("Head Variance Decomposition: Early / Mid / Late", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "layerwise_head_variance_groups.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ── Figure 4: Variance evolution over training epochs ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, run_dir) in zip(axes, RUNS.items()):
        epoch_mads = load_epoch_mads(run_dir)
        if epoch_mads is None:
            ax.text(0.5, 0.5, "No epoch data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(name.capitalize())
            continue
        E, L, H = epoch_mads.shape
        # Var across heads per layer per epoch
        var_over_epochs = np.var(epoch_mads, axis=2)  # (E, L)
        epochs = np.arange(1, E + 1)
        for li in range(L):
            alpha = 0.3 + 0.7 * (li / max(L - 1, 1))
            ax.plot(epochs, var_over_epochs[:, li], label=f"L{li}" if li % 3 == 0 else None,
                    alpha=alpha, linewidth=1.2)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Var(MAD)", fontsize=11)
        ax.set_title(f"{name.capitalize()}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Head Variance Evolution Over Training", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "head_variance_training_evolution.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ── Print numerical summary ──
    print("\n" + "=" * 70)
    print("LAYER-WISE HEAD MAD VARIANCE")
    print("=" * 70)
    header = f"{'Layer':<8}" + "".join(f"{name:>12}" for name in all_mads.keys())
    print(header)
    print("-" * len(header))
    for i, l in enumerate(layers):
        row = f"L{l:<7}"
        for name, mad_mat in all_mads.items():
            v = np.var(mad_mat[i])
            row += f"{v:>12.6f}"
        print(row)
    print("-" * len(header))
    row = f"{'Mean':<8}"
    for name, mad_mat in all_mads.items():
        mean_v = np.mean(np.var(mad_mat, axis=1))
        row += f"{mean_v:>12.6f}"
    print(row)
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# C. HEAD ORDERING PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_head_ordering(all_mads, layers, out_dir):
    """For each layer, sort heads by MAD and compare orderings across runs.

    Three visualisations:
    1. Per-layer sorted MAD bar charts (all runs side-by-side)
    2. Head rank comparison heatmap (which head is rank-1, rank-2, etc.)
    3. Rank stability: do heads keep the same ordering across runs?
    """
    os.makedirs(out_dir, exist_ok=True)
    num_layers = len(layers)
    num_heads = all_mads[next(iter(all_mads))].shape[1]
    run_names = list(all_mads.keys())

    # ── Figure 1: Per-layer sorted MAD values ──
    ncols = 4
    nrows = (num_layers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=False)
    axes_flat = axes.flatten()

    for li, l in enumerate(layers):
        ax = axes_flat[li]
        x_positions = np.arange(num_heads)
        width = 0.25
        offsets_local = {run_names[0]: -width, run_names[1]: 0, run_names[2]: width}

        for name in run_names:
            mad_vals = all_mads[name][li]
            sorted_indices = np.argsort(mad_vals)
            sorted_vals = mad_vals[sorted_indices]
            labels = [f"H{i}" for i in sorted_indices]
            bars = ax.bar(x_positions + offsets_local[name], sorted_vals, width,
                         label=name.capitalize(), color=COLORS[name], alpha=0.85,
                         edgecolor="white", linewidth=0.5)

        # Use baseline sort order for x-labels
        base_sorted = np.argsort(all_mads["baseline"][li])
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"H{i}" for i in base_sorted], fontsize=8)
        ax.set_title(f"Layer {l}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        if li % ncols == 0:
            ax.set_ylabel("MAD", fontsize=10)
        if li == 0:
            ax.legend(fontsize=8)

    # Hide empty subplots
    for i in range(num_layers, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle("Heads Sorted by MAD (Baseline Order) — All Runs", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "head_ordering_sorted_bars.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ── Figure 2: Head rank heatmaps (one per run) ──
    fig, axes = plt.subplots(1, len(run_names), figsize=(7 * len(run_names), 8))
    if len(run_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, run_names):
        # rank_mat[layer, head] = rank of that head (0=smallest MAD, 5=largest)
        rank_mat = np.zeros((num_layers, num_heads), dtype=int)
        for li in range(num_layers):
            rank_mat[li] = np.argsort(np.argsort(all_mads[name][li]))

        im = ax.imshow(rank_mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=num_heads - 1)
        ax.set_xticks(range(num_heads))
        ax.set_xticklabels([f"H{h}" for h in range(num_heads)])
        ax.set_yticks(range(num_layers))
        ax.set_yticklabels([f"L{l}" for l in layers])
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(f"{name.capitalize()} — Head Rank by MAD", fontsize=12, fontweight="bold")
        for i in range(num_layers):
            for j in range(num_heads):
                ax.text(j, i, str(rank_mat[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if rank_mat[i, j] > num_heads // 2 else "black")
    plt.colorbar(im, ax=axes, label="Rank (0=most local, 5=most global)", shrink=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "head_rank_heatmaps.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ── Figure 3: Rank difference from baseline ──
    base_ranks = np.zeros((num_layers, num_heads), dtype=int)
    for li in range(num_layers):
        base_ranks[li] = np.argsort(np.argsort(all_mads["baseline"][li]))

    fig, axes = plt.subplots(1, len(run_names) - 1, figsize=(8 * (len(run_names) - 1), 8))
    if len(run_names) - 1 == 1:
        axes = [axes]

    for ax, name in zip(axes, [n for n in run_names if n != "baseline"]):
        other_ranks = np.zeros((num_layers, num_heads), dtype=int)
        for li in range(num_layers):
            other_ranks[li] = np.argsort(np.argsort(all_mads[name][li]))
        diff = other_ranks - base_ranks
        vmax = max(abs(diff.min()), abs(diff.max()), 1)
        im = ax.imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(num_heads))
        ax.set_xticklabels([f"H{h}" for h in range(num_heads)])
        ax.set_yticks(range(num_layers))
        ax.set_yticklabels([f"L{l}" for l in layers])
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(f"Rank Shift: {name.capitalize()} − Baseline", fontsize=12, fontweight="bold")
        for i in range(num_layers):
            for j in range(num_heads):
                ax.text(j, i, f"{diff[i, j]:+d}", ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax, label="Rank change", shrink=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "head_rank_shift_from_baseline.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ── Figure 4: Per-layer MAD profile (connected dots, sorted) ──
    fig, axes = plt.subplots(3, 4, figsize=(20, 14), sharey=True)
    axes_flat = axes.flatten()

    for li, l in enumerate(layers):
        ax = axes_flat[li]
        for name in run_names:
            mad_vals = np.sort(all_mads[name][li])
            ax.plot(range(num_heads), mad_vals, marker=MARKERS[name],
                   label=name.capitalize(), color=COLORS[name],
                   linewidth=2, markersize=7)
        ax.set_xticks(range(num_heads))
        ax.set_xticklabels([f"Rank {i}" for i in range(num_heads)], fontsize=8)
        ax.set_title(f"Layer {l}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if li == 0:
            ax.legend(fontsize=9)
        if li % 4 == 0:
            ax.set_ylabel("MAD (sorted)", fontsize=10)

    for i in range(num_layers, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle("Sorted Head MAD Profiles: Specialisation Emergence", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "head_ordering_profiles.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ── Figure 5: Gap between most-local and most-global head per layer ──
    fig, ax = plt.subplots(figsize=(12, 5))
    for name in run_names:
        gaps = []
        for li in range(num_layers):
            sorted_m = np.sort(all_mads[name][li])
            gaps.append(sorted_m[-1] - sorted_m[0])
        ax.plot(layers, gaps, marker=MARKERS[name], label=name.capitalize(),
               color=COLORS[name], linewidth=2, markersize=8)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("MAD gap (max − min head)", fontsize=12)
    ax.set_title("Local–Global Head Separation by Layer", fontsize=14, fontweight="bold")
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "head_mad_gap.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ── Figure 6: Spearman rank correlation of head orderings across runs ──
    from scipy.stats import spearmanr
    pairs = [(a, b) for i, a in enumerate(run_names) for b in run_names[i+1:]]
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5))
    if len(pairs) == 1:
        axes = [axes]

    for ax, (name_a, name_b) in zip(axes, pairs):
        correlations = []
        for li in range(num_layers):
            rho, _ = spearmanr(all_mads[name_a][li], all_mads[name_b][li])
            correlations.append(rho)
        ax.bar(range(num_layers), correlations, color=COLORS.get(name_b, "gray"), alpha=0.8)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xticks(range(num_layers))
        ax.set_xticklabels([f"L{l}" for l in layers])
        ax.set_xlabel("Layer")
        ax.set_ylabel("Spearman ρ")
        ax.set_title(f"{name_a.capitalize()} vs {name_b.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_ylim(-0.2, 1.1)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Head Ordering Stability (Spearman Rank Correlation)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "head_rank_spearman.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ── Print numerical summary ──
    print("\n" + "=" * 70)
    print("HEAD ORDERING ANALYSIS")
    print("=" * 70)
    for li, l in enumerate(layers):
        print(f"\nLayer {l}:")
        for name in run_names:
            sorted_idx = np.argsort(all_mads[name][li])
            sorted_vals = all_mads[name][li][sorted_idx]
            ordering = " < ".join([f"H{i}({v:.3f})" for i, v in zip(sorted_idx, sorted_vals)])
            print(f"  {name:>10}: {ordering}")
    print()
    print("Spearman rank correlations:")
    for name_a, name_b in pairs:
        for li, l in enumerate(layers):
            rho, p = spearmanr(all_mads[name_a][li], all_mads[name_b][li])
            print(f"  L{l}: {name_a} vs {name_b}: ρ={rho:.3f} (p={p:.4f})")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Head variance & ordering analysis")
    parser.add_argument("--output_dir", type=str, default="visualizations/head_analysis")
    args = parser.parse_args()

    print("Loading evaluation results...")
    all_mads = {}
    layers = None
    for name, run_dir in RUNS.items():
        eval_data = load_eval(run_dir)
        mad_mat, layer_ids = extract_mad_matrix(eval_data)
        all_mads[name] = mad_mat
        if layers is None:
            layers = layer_ids
        print(f"  {name}: {mad_mat.shape} (layers × heads)")

    print(f"\nGenerating plots in {args.output_dir}/...")
    plot_layerwise_variance(all_mads, layers, args.output_dir)
    print("  ✓ Layer-wise head variance plots")

    plot_head_ordering(all_mads, layers, args.output_dir)
    print("  ✓ Head ordering plots")

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
