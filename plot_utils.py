"""Visualization utilities for the bimodal head specialization experiment."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_mad_heatmap(mad_dict, save_path, title="Mean Attention Distance by Layer × Head"):
    """Plot layer × head heatmap of MAD values.

    Args:
        mad_dict: {block_idx: np.array of shape (num_heads,)}
        save_path: path to save the figure
        title: figure title
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    block_indices = sorted(mad_dict.keys())
    num_blocks = len(block_indices)
    num_heads = len(next(iter(mad_dict.values())))

    data = np.zeros((num_blocks, num_heads))
    for i, bidx in enumerate(block_indices):
        data[i] = mad_dict[bidx]

    fig, ax = plt.subplots(figsize=(max(8, num_heads), max(6, num_blocks * 0.6)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlBu_r")
    ax.set_xticks(range(num_heads))
    ax.set_xticklabels([f"H{h}" for h in range(num_heads)])
    ax.set_yticks(range(num_blocks))
    ax.set_yticklabels([f"Block {bidx}" for bidx in block_indices])
    ax.set_xlabel("Head")
    ax.set_ylabel("Block")
    ax.set_title(title)

    # Add text annotations
    for i in range(num_blocks):
        for j in range(num_heads):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if data[i, j] > data.mean() else "black")

    plt.colorbar(im, ax=ax, label="MAD")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_mad_violins(mad_per_image_dict, save_path, block_indices=None, title="MAD Distribution per Block"):
    """Violin plot of head MADs across images for each block.

    Args:
        mad_per_image_dict: {block_idx: np.array of shape (num_images, num_heads)}
        save_path: path to save
        block_indices: which blocks to plot (default: all)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if block_indices is None:
        block_indices = sorted(mad_per_image_dict.keys())

    n_blocks = len(block_indices)
    fig, axes = plt.subplots(1, n_blocks, figsize=(3 * n_blocks, 5), sharey=True)
    if n_blocks == 1:
        axes = [axes]

    for ax, bidx in zip(axes, block_indices):
        data = mad_per_image_dict[bidx]  # (num_images, num_heads)
        num_heads = data.shape[1]
        parts = ax.violinplot([data[:, h] for h in range(num_heads)],
                              positions=range(num_heads), showmeans=True, showmedians=True)
        ax.set_title(f"Block {bidx}")
        ax.set_xlabel("Head")
        ax.set_xticks(range(num_heads))
        ax.set_xticklabels([f"H{h}" for h in range(num_heads)])
        if ax == axes[0]:
            ax.set_ylabel("MAD")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison_heatmaps(baseline_mad, regularized_mad, save_path):
    """Side-by-side MAD heatmaps for baseline vs regularized."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    block_indices = sorted(set(baseline_mad.keys()) & set(regularized_mad.keys()))
    num_blocks = len(block_indices)
    num_heads = len(next(iter(baseline_mad.values())))

    data_b = np.zeros((num_blocks, num_heads))
    data_r = np.zeros((num_blocks, num_heads))
    for i, bidx in enumerate(block_indices):
        data_b[i] = baseline_mad[bidx]
        data_r[i] = regularized_mad[bidx]

    vmin = min(data_b.min(), data_r.min())
    vmax = max(data_b.max(), data_r.max())

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, max(6, num_blocks * 0.6)),
                                         gridspec_kw={"width_ratios": [1, 1, 1]})

    for ax, data, title in [(ax1, data_b, "Baseline FT"), (ax2, data_r, "Regularized FT")]:
        im = ax.imshow(data, aspect="auto", cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(num_heads))
        ax.set_xticklabels([f"H{h}" for h in range(num_heads)])
        ax.set_yticks(range(num_blocks))
        ax.set_yticklabels([f"Block {bidx}" for bidx in block_indices])
        ax.set_xlabel("Head")
        ax.set_ylabel("Block")
        ax.set_title(title)
        for i in range(num_blocks):
            for j in range(num_heads):
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if data[i, j] > (vmin + vmax) / 2 else "black")

    # Difference heatmap
    diff = data_r - data_b
    im3 = ax3.imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-abs(diff).max(), vmax=abs(diff).max())
    ax3.set_xticks(range(num_heads))
    ax3.set_xticklabels([f"H{h}" for h in range(num_heads)])
    ax3.set_yticks(range(num_blocks))
    ax3.set_yticklabels([f"Block {bidx}" for bidx in block_indices])
    ax3.set_xlabel("Head")
    ax3.set_ylabel("Block")
    ax3.set_title("Difference (Reg − Base)")
    for i in range(num_blocks):
        for j in range(num_heads):
            ax3.text(j, i, f"{diff[i, j]:+.3f}", ha="center", va="center", fontsize=7)

    plt.colorbar(im, ax=ax2, label="MAD")
    plt.colorbar(im3, ax=ax3, label="ΔMAD")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(log_path, save_path):
    """Plot training curves from a JSON log file."""
    import json
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(log_path) as f:
        logs = json.load(f)

    epochs = [e["epoch"] for e in logs]
    train_loss = [e["train_loss"] for e in logs]
    val_acc1 = [e["val_acc1"] for e in logs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_loss, "b-o", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_acc1, "r-o", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Top-1 (%)")
    ax2.set_title("Validation Accuracy")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_head_masking_results(results, save_path):
    """Bar chart of accuracy drops under different head masking strategies.

    Args:
        results: dict with keys 'no_mask', 'mask_local', 'mask_global', 'mask_random'
                 each mapping to accuracy values.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    labels = ["No Mask", "Mask Local", "Mask Global", "Mask Random"]
    baseline_vals = [results["baseline"][k] for k in ["no_mask", "mask_local", "mask_global", "mask_random"]]
    reg_vals = [results["regularized"][k] for k in ["no_mask", "mask_local", "mask_global", "mask_random"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, baseline_vals, width, label="Baseline FT", color="steelblue")
    bars2 = ax.bar(x + width / 2, reg_vals, width, label="Regularized FT", color="coral")

    ax.set_ylabel("Val Top-1 (%)")
    ax.set_title("Head Masking Experiment")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─── Segmentation-Specific Visualizations ────────────────────────────────────

def plot_seg_training_curves(log_path, save_path):
    """Plot training curves: loss, mIoU, boundary F1, regularizer loss."""
    import json
    with open(log_path) as f:
        logs = json.load(f)

    epochs = [l["epoch"] for l in logs]
    task_loss = [l["train_loss"] for l in logs]
    miou = [l["val_miou"] for l in logs]
    bf1 = [l["val_boundary_f1"] for l in logs]
    has_reg = any(l.get("reg_loss", 0) > 0 for l in logs)

    ncols = 4 if has_reg else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))

    axes[0].plot(epochs, task_loss, "b-")
    axes[0].set_title("Task Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, miou, "g-")
    axes[1].set_title("Val mIoU")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, bf1, "r-")
    axes[2].set_title("Val Boundary F1")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)

    if has_reg:
        reg_loss = [l.get("reg_loss", 0) for l in logs]
        axes[3].plot(epochs, reg_loss, "m-")
        axes[3].set_title("Regularizer Loss")
        axes[3].set_xlabel("Epoch")
        axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_seg_head_masking(baseline_masking, regularized_masking, save_path):
    """Bar chart: mIoU and boundary F1 under different masking conditions."""
    conditions = ["no_mask", "mask_local", "mask_global", "mask_random"]
    labels = ["No Mask", "Mask Local", "Mask Global", "Mask Random"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(conditions))
    w = 0.35

    for ax, metric, title in [
        (axes[0], "miou", "mIoU"),
        (axes[1], "boundary_f1", "Boundary F1"),
    ]:
        base_vals = [baseline_masking[c][metric] for c in conditions]
        reg_vals = [regularized_masking[c][metric] for c in conditions]
        b1 = ax.bar(x - w / 2, base_vals, w, label="Baseline", color="steelblue")
        b2 = ax.bar(x + w / 2, reg_vals, w, label="Regularized", color="coral")
        ax.set_title(f"Head Masking: {title}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        for bars in [b1, b2]:
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_distance_histograms(base_hist, reg_hist, block_idx, save_path):
    """Side-by-side distance histograms for each head in a block.

    Args:
        base_hist: (num_heads, num_bins) array for baseline
        reg_hist: (num_heads, num_bins) for regularized
        block_idx: block index (for title)
    """
    num_heads = base_hist.shape[0]
    num_bins = base_hist.shape[1]
    fig, axes = plt.subplots(2, num_heads, figsize=(3 * num_heads, 6), sharey=True)
    bin_centers = np.arange(num_bins)

    for h in range(num_heads):
        axes[0, h].bar(bin_centers, base_hist[h], color="steelblue", alpha=0.8)
        axes[0, h].set_title(f"Head {h}")
        if h == 0:
            axes[0, h].set_ylabel("Baseline")

        axes[1, h].bar(bin_centers, reg_hist[h], color="coral", alpha=0.8)
        if h == 0:
            axes[1, h].set_ylabel("Regularized")
        axes[1, h].set_xlabel("Dist. bin")

    fig.suptitle(f"Attention Distance Histograms — Block {block_idx}", fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_conditional_mad(baseline_cond, regularized_cond, save_path):
    """Grouped bar chart: boundary vs interior MAD per head, baseline vs regularized."""
    blocks = sorted([int(k) for k in baseline_cond.keys()])
    num_heads = len(baseline_cond[str(blocks[0])]["boundary_mad"])

    fig, axes = plt.subplots(len(blocks), 1, figsize=(10, 3.5 * len(blocks)))
    if len(blocks) == 1:
        axes = [axes]

    for ax, bidx in zip(axes, blocks):
        bb = np.array(baseline_cond[str(bidx)]["boundary_mad"])
        bi = np.array(baseline_cond[str(bidx)]["interior_mad"])
        rb = np.array(regularized_cond[str(bidx)]["boundary_mad"])
        ri = np.array(regularized_cond[str(bidx)]["interior_mad"])

        x = np.arange(num_heads)
        w = 0.2
        ax.bar(x - 1.5 * w, bb, w, label="Base boundary", color="steelblue")
        ax.bar(x - 0.5 * w, bi, w, label="Base interior", color="lightblue")
        ax.bar(x + 0.5 * w, rb, w, label="Reg boundary", color="coral")
        ax.bar(x + 1.5 * w, ri, w, label="Reg interior", color="lightsalmon")
        ax.set_title(f"Block {bidx}: Boundary vs Interior MAD")
        ax.set_xlabel("Head")
        ax.set_ylabel("MAD")
        ax.set_xticks(x)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
