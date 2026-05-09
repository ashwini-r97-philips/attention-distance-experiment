"""Visualisation utilities for attention-distance experiments."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ─── 1. Training curves ─────────────────────────────────────────────────────

def plot_training_curves(logs, save_path, title_prefix=""):
    """Plot train loss, val loss, top-1, top-5, lr, and reg loss vs epoch."""
    _ensure_dir(save_path)
    epochs = [e["epoch"] for e in logs]

    has_reg = any(e.get("reg_loss", 0) > 0 for e in logs)
    ncols = 6 if has_reg else 5
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4))

    def _plot(ax, key, label, color="b"):
        vals = [e.get(key, 0) for e in logs]
        ax.plot(epochs, vals, f"{color}-", linewidth=1.2)
        ax.set_xlabel("Epoch")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    _plot(axes[0], "train_loss", f"{title_prefix}Train Loss")
    _plot(axes[1], "val_loss", f"{title_prefix}Val Loss", "r")
    _plot(axes[2], "val_acc1", f"{title_prefix}Top-1 %", "g")
    _plot(axes[3], "val_acc5", f"{title_prefix}Top-5 %", "m")
    _plot(axes[4], "lr", f"{title_prefix}Learning Rate", "orange")
    if has_reg:
        _plot(axes[5], "reg_loss", f"{title_prefix}Reg Loss", "purple")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─── 2. MAD heatmap ─────────────────────────────────────────────────────────

def plot_mad_heatmap(mad_dict, save_path, title="MAD by Layer × Head"):
    """Layer × head heatmap of MAD values.

    Args:
        mad_dict: {block_idx: np.array of shape (num_heads,)}
    """
    _ensure_dir(save_path)
    blocks = sorted(mad_dict.keys())
    num_blocks = len(blocks)
    num_heads = len(next(iter(mad_dict.values())))

    data = np.zeros((num_blocks, num_heads))
    for i, b in enumerate(blocks):
        data[i] = mad_dict[b]

    fig, ax = plt.subplots(figsize=(max(8, num_heads), max(6, num_blocks * 0.6)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlBu_r")
    ax.set_xticks(range(num_heads))
    ax.set_xticklabels([f"H{h}" for h in range(num_heads)])
    ax.set_yticks(range(num_blocks))
    ax.set_yticklabels([f"L{b}" for b in blocks])
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    for i in range(num_blocks):
        for j in range(num_heads):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if data[i, j] > data.mean() else "black")
    plt.colorbar(im, ax=ax, label="MAD")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison_heatmaps(base_mad, tgt_mad, save_path,
                             title_a="Baseline", title_b="Targeted"):
    """Side-by-side MAD heatmaps + difference."""
    _ensure_dir(save_path)
    blocks = sorted(set(base_mad.keys()) & set(tgt_mad.keys()))
    num_blocks = len(blocks)
    num_heads = len(next(iter(base_mad.values())))

    data_b = np.array([base_mad[b] for b in blocks])
    data_t = np.array([tgt_mad[b] for b in blocks])

    vmin = min(data_b.min(), data_t.min())
    vmax = max(data_b.max(), data_t.max())

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, max(6, num_blocks * 0.6)),
                                         gridspec_kw={"width_ratios": [1, 1, 1]})
    for ax, data, title in [(ax1, data_b, title_a), (ax2, data_t, title_b)]:
        im = ax.imshow(data, aspect="auto", cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(num_heads))
        ax.set_xticklabels([f"H{h}" for h in range(num_heads)])
        ax.set_yticks(range(num_blocks))
        ax.set_yticklabels([f"L{b}" for b in blocks])
        ax.set_xlabel("Head"); ax.set_ylabel("Layer"); ax.set_title(title)
        for i in range(num_blocks):
            for j in range(num_heads):
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if data[i, j] > (vmin + vmax) / 2 else "black")

    diff = data_t - data_b
    im3 = ax3.imshow(diff, aspect="auto", cmap="RdBu_r",
                     vmin=-abs(diff).max(), vmax=abs(diff).max())
    ax3.set_xticks(range(num_heads))
    ax3.set_xticklabels([f"H{h}" for h in range(num_heads)])
    ax3.set_yticks(range(num_blocks))
    ax3.set_yticklabels([f"L{b}" for b in blocks])
    ax3.set_xlabel("Head"); ax3.set_ylabel("Layer")
    ax3.set_title(f"Δ ({title_b} − {title_a})")
    for i in range(num_blocks):
        for j in range(num_heads):
            ax3.text(j, i, f"{diff[i, j]:+.3f}", ha="center", va="center", fontsize=7)

    plt.colorbar(im, ax=ax2, label="MAD")
    plt.colorbar(im3, ax=ax3, label="ΔMAD")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─── 3. MAD distribution by layer ───────────────────────────────────────────

def plot_mad_distributions(base_mad, tgt_mad, save_path, blocks=None):
    """Per-layer histogram of headwise MAD: baseline vs targeted."""
    _ensure_dir(save_path)
    if blocks is None:
        blocks = sorted(set(base_mad.keys()) & set(tgt_mad.keys()))
    n = len(blocks)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, b in zip(axes, blocks):
        bv = np.array(base_mad[b])
        tv = np.array(tgt_mad[b])
        bins = np.linspace(0, 1, 20)
        ax.hist(bv, bins=bins, alpha=0.5, label="Baseline", color="steelblue")
        ax.hist(tv, bins=bins, alpha=0.5, label="Targeted", color="coral")
        ax.set_title(f"Layer {b}")
        ax.set_xlabel("MAD")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Count")
    fig.suptitle("Head MAD Distribution by Layer", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─── 4. Headwise MAD trajectory over training ───────────────────────────────

def plot_mad_trajectories(epoch_mads, save_path, layers=None, title_prefix=""):
    """Plot per-head MAD over epochs for selected layers.

    Args:
        epoch_mads: list of dicts, each {layer_idx: [h0, h1, ...]}
    """
    _ensure_dir(save_path)
    all_layers = sorted(epoch_mads[0].keys()) if epoch_mads else []
    if layers is None:
        layers = all_layers[:4] + all_layers[-2:]  # early + late
        layers = sorted(set(layers))
    n = len(layers)
    if n == 0:
        return

    epochs = list(range(1, len(epoch_mads) + 1))
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        num_heads = len(epoch_mads[0][layer])
        for h in range(num_heads):
            vals = [epoch_mads[e][layer][h] for e in range(len(epoch_mads))]
            ax.plot(epochs, vals, label=f"H{h}", linewidth=1)
        ax.set_title(f"{title_prefix}Layer {layer}")
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("MAD")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─── 5. Inter-head MAD variance by layer ────────────────────────────────────

def plot_mad_variance(base_mad, tgt_mad, save_path):
    """Bar chart: Var_h(MAD) per layer, baseline vs targeted."""
    _ensure_dir(save_path)
    blocks = sorted(set(base_mad.keys()) & set(tgt_mad.keys()))
    b_var = [np.var(base_mad[b]) for b in blocks]
    t_var = [np.var(tgt_mad[b]) for b in blocks]
    x = np.arange(len(blocks))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, b_var, w, label="Baseline", color="steelblue")
    ax.bar(x + w / 2, t_var, w, label="Targeted", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{b}" for b in blocks])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Var(MAD)")
    ax.set_title("Inter-Head MAD Variance by Layer")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─── 6. Local mass heatmaps ─────────────────────────────────────────────────

def plot_local_mass_heatmaps(base_lm, tgt_lm, tau, save_path):
    """Side-by-side local-mass heatmaps for a given tau."""
    plot_comparison_heatmaps(
        base_lm, tgt_lm, save_path,
        title_a=f"Baseline (τ={tau})", title_b=f"Targeted (τ={tau})",
    )


# ─── 7. Entropy heatmap ─────────────────────────────────────────────────────

def plot_entropy_heatmap(base_ent, tgt_ent, save_path):
    plot_comparison_heatmaps(
        base_ent, tgt_ent, save_path,
        title_a="Baseline Entropy", title_b="Targeted Entropy",
    )


# ─── 8. Distance histogram ──────────────────────────────────────────────────

def plot_distance_histograms(base_hist, tgt_hist, block_idx, save_path):
    """Per-head distance histogram for one layer."""
    _ensure_dir(save_path)
    num_heads = base_hist.shape[0]
    num_bins = base_hist.shape[1]
    fig, axes = plt.subplots(2, num_heads, figsize=(3 * num_heads, 6), sharey=True)
    bins = np.arange(num_bins)
    for h in range(num_heads):
        axes[0, h].bar(bins, base_hist[h], color="steelblue", alpha=0.8)
        axes[0, h].set_title(f"H{h}")
        if h == 0:
            axes[0, h].set_ylabel("Baseline")
        axes[1, h].bar(bins, tgt_hist[h], color="coral", alpha=0.8)
        if h == 0:
            axes[1, h].set_ylabel("Targeted")
        axes[1, h].set_xlabel("Dist bin")
    fig.suptitle(f"Attention Distance Histograms — Layer {block_idx}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─── 9. Attention map visualisation ─────────────────────────────────────────

def plot_attention_maps(images, attn_maps, query_patches, head_labels,
                        save_path, grid_h=14, grid_w=14):
    """Visualise attention maps for selected query patches.

    Args:
        images: list of (C, H, W) tensors (unnormalised for display).
        attn_maps: list of (H_heads, N, N) arrays (patch-only attention).
        query_patches: list of patch indices.
        head_labels: list of (head_idx, layer_name) tuples to display.
        grid_h, grid_w: patch grid dims.
    """
    _ensure_dir(save_path)
    n_imgs = len(images)
    n_heads = len(head_labels)
    n_queries = len(query_patches)
    fig, axes = plt.subplots(n_imgs, n_heads * n_queries + 1,
                             figsize=(2.5 * (n_heads * n_queries + 1), 2.5 * n_imgs))
    if n_imgs == 1:
        axes = axes[np.newaxis, :]

    for img_idx in range(n_imgs):
        img = images[img_idx]
        if hasattr(img, "numpy"):
            img = img.numpy()
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes[img_idx, 0].imshow(img)
        axes[img_idx, 0].set_title("Input", fontsize=8)
        axes[img_idx, 0].axis("off")

        col = 1
        for q in query_patches:
            for hi, (head_idx, lbl) in enumerate(head_labels):
                attn = attn_maps[img_idx][head_idx]  # (N, N)
                attn_q = attn[q].reshape(grid_h, grid_w)
                axes[img_idx, col].imshow(attn_q, cmap="hot", vmin=0)
                axes[img_idx, col].set_title(f"{lbl} q={q}", fontsize=6)
                axes[img_idx, col].axis("off")
                col += 1

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─── 11. Bimodality diagnostics ─────────────────────────────────────────────

def plot_bimodality_histogram(all_mads, save_path, title="MAD Distribution"):
    """Histogram of all d_lh values, optionally split by layer group."""
    _ensure_dir(save_path)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    bins = np.linspace(0, 1, 30)

    # All
    axes[0].hist(all_mads, bins=bins, color="steelblue", alpha=0.8)
    axes[0].set_title("All Layers")
    axes[0].set_xlabel("MAD")

    n = len(all_mads)
    num_heads = 6  # default
    num_layers = n // num_heads if num_heads > 0 else 12
    third = num_layers // 3

    early = all_mads[:third * num_heads]
    mid = all_mads[third * num_heads:2 * third * num_heads]
    late = all_mads[2 * third * num_heads:]

    for ax, data, lbl in [(axes[1], early, "Early"), (axes[2], mid, "Mid"), (axes[3], late, "Late")]:
        if len(data) > 0:
            ax.hist(data, bins=bins, color="coral", alpha=0.8)
        ax.set_title(lbl)
        ax.set_xlabel("MAD")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─── 12. Summary table ──────────────────────────────────────────────────────

def save_summary_table(baseline_stats, targeted_stats, save_dir):
    """Save CSV and markdown comparison table."""
    _ensure_dir(os.path.join(save_dir, "dummy"))
    rows = []
    for key in ["top1", "top5", "mean_mad", "mean_mad_variance",
                "mean_local_mass_0.15", "mean_local_mass_0.25", "mean_local_mass_0.35",
                "mean_entropy"]:
        bv = baseline_stats.get(key, "—")
        tv = targeted_stats.get(key, "—")
        rows.append((key, bv, tv))

    # CSV
    csv_path = os.path.join(save_dir, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("metric,baseline,targeted\n")
        for k, bv, tv in rows:
            f.write(f"{k},{bv},{tv}\n")

    # Markdown
    md_path = os.path.join(save_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("| Metric | Baseline | Targeted |\n")
        f.write("|--------|--------:|--------:|\n")
        for k, bv, tv in rows:
            f.write(f"| {k} | {bv} | {tv} |\n")
