"""Visualise learned gate fields over CUB images.

For each image shows a grid of 14×14 heatmaps — one per (block, head) — where
each cell colour is the gate value for that patch token. Positive (warm) = local
bias, negative (cool) = global bias, zero = no change.

For the scalar gate (v3): gate value is a single scalar per patch per head.
For the vector gate (v3-vec): gate value is collapsed to a scalar by computing
  the effective penalty at the median pairwise distance:
  scalar[i,h] = sum_k gate_vec[i,h,k] * rbf_k(median_dist)

Usage
-----
  # Scalar gate checkpoint
  python visualize_gate_fields.py \
      --checkpoint runs/cub/token_v3_50ep_cub_vit_s16/checkpoints/best.pth \
      --config    configs/cub/token_v3_50ep_cub_vit_s16.yaml \
      --n_images  8 \
      --output_dir vis/gate_fields_v3

  # Vector gate checkpoint
  python visualize_gate_fields.py \
      --checkpoint runs/cub/token_v3_vec_50ep_cub_vit_s16/checkpoints/best.pth \
      --config    configs/cub/token_v3_vec_50ep_cub_vit_s16.yaml \
      --n_images  8 \
      --output_dir vis/gate_fields_vec
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "bimodal_head_specialisation"))
sys.path.insert(0, _HERE)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image as PILImage
from torchvision import transforms
from datasets import load_dataset

from common.config import load_config
from common.model_utils import load_vit_small
from token_locality_gate_v3 import (
    TokenLocalityGateModuleV3,
    VectorKernelGateModule,
    RBFDistanceEncoder,
)


# ─── Gate field extraction ────────────────────────────────────────────────────

def extract_scalar_gate_fields(model, gate_module, images, device):
    """Extract per-patch per-head scalar gate values for v3.

    Returns: dict[block_idx] → np.ndarray (B, Np, H)
    """
    fields = {}
    hooks = []

    for i, block_idx in enumerate(gate_module.block_indices):
        branch = gate_module.gates[i]
        captured = {}

        def make_hook(cap, br):
            def hook(module, inp, out):
                x = inp[0]
                with torch.no_grad():
                    g = br.gate_scale * torch.tanh(br.linear(x))  # (B, N, H)
                cap["gate"] = g[:, 1:, :].cpu().float()            # (B, Np, H)
            return hook

        h = model.blocks[block_idx].norm1.register_forward_hook(
            make_hook(captured, branch)
        )
        hooks.append((h, captured, block_idx))

    model.eval()
    with torch.no_grad():
        _ = model(images.to(device))

    for h, captured, block_idx in hooks:
        h.remove()
        if "gate" in captured:
            fields[block_idx] = captured["gate"].numpy()

    return fields


def extract_vector_gate_fields(model, gate_module, images, device):
    """Extract per-patch per-head scalar gate values for v3-vec.

    Collapses K-dim vector to scalar via dot with RBF features at median dist.

    Returns: dict[block_idx] → np.ndarray (B, Np, H)
    """
    dist_mat = gate_module.dist_matrix.cpu().float()
    Np = dist_mat.shape[0]
    upper = dist_mat[torch.triu(torch.ones(Np, Np, dtype=torch.bool), diagonal=1)]
    median_dist = upper.median().item()

    K = gate_module.rbf_features.shape[-1]
    rbf_enc = RBFDistanceEncoder(num_basis=K)
    rbf_at_median = rbf_enc(torch.tensor([[median_dist]]))[0, 0, :].float()  # (K,)

    fields = {}
    hooks = []

    for i, block_idx in enumerate(gate_module.block_indices):
        branch = gate_module.gates[i]
        captured = {}

        def make_hook(cap, br, rbf_vec):
            def hook(module, inp, out):
                x = inp[0]
                B, N, D = x.shape
                with torch.no_grad():
                    raw = br.gate_scale * torch.tanh(br.mlp(x))   # (B, N, H*K)
                    vec = raw.view(B, N, br.num_heads, br.num_basis)
                    vec = vec[:, 1:, :, :]                          # (B, Np, H, K)
                    scalar = (vec * rbf_vec.to(vec.device)).sum(-1) # (B, Np, H)
                cap["gate"] = scalar.cpu().float()
            return hook

        h = model.blocks[block_idx].norm1.register_forward_hook(
            make_hook(captured, branch, rbf_at_median)
        )
        hooks.append((h, captured, block_idx))

    model.eval()
    with torch.no_grad():
        _ = model(images.to(device))

    for h, captured, block_idx in hooks:
        h.remove()
        if "gate" in captured:
            fields[block_idx] = captured["gate"].numpy()

    return fields


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def unnormalise(tensor):
    """Reverse ImageNet normalisation → HWC float32 in [0,1]."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).numpy() * std + mean
    return np.clip(img, 0, 1)


def upsample_gate(gate_2d, size=224):
    """Bilinear upsample a (G, G) gate map to (size, size)."""
    arr = ((gate_2d - gate_2d.min()) /
           max(gate_2d.max() - gate_2d.min(), 1e-6) * 255).astype(np.uint8)
    return np.array(PILImage.fromarray(arr).resize((size, size),
                                                    PILImage.BILINEAR)) / 255.0


# ─── Plot 1: per-image heatmap grid (block × head) ───────────────────────────

def plot_gate_grid(img_tensor, fields, block_indices, num_heads,
                   grid_size, img_idx, output_dir, gate_type):
    """14×14 heatmap per (block, head) for one image."""
    n_rows = len(block_indices)
    n_cols = num_heads + 1   # col 0 = original image

    fig = plt.figure(figsize=(n_cols * 1.8, n_rows * 1.8 + 0.5))
    fig.suptitle(f"{gate_type} — gate fields — image {img_idx}", fontsize=8)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.05)

    img_np = unnormalise(img_tensor)

    all_vals = np.concatenate([
        fields[b][img_idx].flatten() for b in block_indices if b in fields
    ])
    vmax = max(abs(all_vals.max()), abs(all_vals.min()), 0.01)

    for row, block_idx in enumerate(block_indices):
        ax_img = fig.add_subplot(gs[row, 0])
        ax_img.imshow(img_np)
        ax_img.axis("off")
        ax_img.set_title(f"blk {block_idx}", fontsize=6)

        if block_idx not in fields:
            continue
        gate_map = fields[block_idx][img_idx]   # (Np, H)

        for h in range(num_heads):
            ax = fig.add_subplot(gs[row, h + 1])
            patch_vals = gate_map[:, h].reshape(grid_size, grid_size)
            ax.imshow(patch_vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      interpolation="nearest")
            ax.axis("off")
            if row == 0:
                ax.set_title(f"h{h}", fontsize=6)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                                norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    fig.colorbar(sm, cax=cbar_ax)
    cbar_ax.tick_params(labelsize=6)
    cbar_ax.set_ylabel("local (+) / global (−)", fontsize=6)

    path = os.path.join(output_dir, f"img_{img_idx:03d}_grid.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ─── Plot 2: mean gate field across all images ────────────────────────────────

def plot_mean_gate_field(fields, block_indices, num_heads, grid_size,
                          output_dir, gate_type):
    """Average gate field over all images — shows structural head preferences."""
    n_rows = len(block_indices)
    fig, axes = plt.subplots(n_rows, num_heads,
                             figsize=(num_heads * 1.8, n_rows * 1.8 + 0.4))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"{gate_type} — mean gate field (all images)", fontsize=8)

    all_means = [fields[b].mean(axis=0) for b in block_indices if b in fields]
    if not all_means:
        plt.close(fig)
        return
    vmax = max(abs(np.stack(all_means).max()), abs(np.stack(all_means).min()), 0.01)

    for row, block_idx in enumerate(block_indices):
        if block_idx not in fields:
            continue
        mean_map = fields[block_idx].mean(axis=0)   # (Np, H)
        for h in range(num_heads):
            ax = axes[row, h]
            ax.imshow(mean_map[:, h].reshape(grid_size, grid_size),
                      cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      interpolation="nearest")
            ax.axis("off")
            if row == 0:
                ax.set_title(f"head {h}", fontsize=7)
            if h == 0:
                ax.set_ylabel(f"blk {block_idx}", fontsize=7, rotation=0,
                              labelpad=28)

    path = os.path.join(output_dir, "mean_gate_field.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ─── Plot 3: overlay on image (active heads only) ────────────────────────────

def plot_overlay(img_tensor, fields, block_indices, num_heads, grid_size,
                 img_idx, output_dir, gate_type, threshold=0.05):
    """Overlay gate field on image. Red = local bias, blue = global bias.

    Skips heads whose mean |gate| is below threshold (near-zero heads).
    """
    img_np = unnormalise(img_tensor)

    active = [
        (b, h)
        for b in block_indices if b in fields
        for h in range(num_heads)
        if abs(fields[b][img_idx][:, h].mean()) > threshold
    ]

    if not active:
        print(f"  img {img_idx}: no active heads above threshold={threshold:.3f}")
        return

    ncols = min(len(active), 6)
    nrows = (len(active) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.2, nrows * 2.2 + 0.4))
    axes_flat = np.array(axes).reshape(-1) if len(active) > 1 else [axes]
    fig.suptitle(f"{gate_type} — active gates — image {img_idx}", fontsize=8)

    for ax_idx, (block_idx, h) in enumerate(active):
        gate_vals = fields[block_idx][img_idx][:, h]   # (Np,)
        gate_2d = gate_vals.reshape(grid_size, grid_size)

        # Normalise to [0,1] centred at 0.5
        vmax = max(abs(gate_vals.max()), abs(gate_vals.min()), 1e-4)
        gate_norm_2d = gate_2d / vmax * 0.5 + 0.5   # 0.5 = neutral
        gate_up = np.array(
            PILImage.fromarray((gate_norm_2d * 255).astype(np.uint8)).resize(
                (224, 224), PILImage.BILINEAR
            )
        ) / 255.0   # [0,1]

        alpha = np.abs(gate_up - 0.5) * 1.4
        alpha = np.clip(alpha, 0, 0.65)[:, :, np.newaxis]
        colour = np.zeros_like(img_np)
        colour[:, :, 0] = (gate_up > 0.5).astype(float)   # red = local
        colour[:, :, 2] = (gate_up < 0.5).astype(float)   # blue = global
        blended = np.clip((1 - alpha) * img_np + alpha * colour, 0, 1)

        ax = axes_flat[ax_idx]
        ax.imshow(blended)
        ax.axis("off")
        ax.set_title(f"blk{block_idx} h{h}  {gate_vals.mean():+.3f}", fontsize=7)

    for ax in axes_flat[len(active):]:
        ax.axis("off")

    path = os.path.join(output_dir, f"img_{img_idx:03d}_overlay.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n_images", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="vis/gate_fields")
    parser.add_argument("--head_threshold", type=float, default=0.05,
                        help="Min mean |gate| to include head in overlay plots")
    parser.add_argument("--image_indices", type=int, nargs="+", default=None,
                        help="Specific test image indices (overrides --n_images)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Model ──
    print(f"Loading model from {args.checkpoint}...")
    model = load_vit_small(cfg, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    # ── Gate module ──
    gate_type = cfg.reg_type
    if gate_type == "token_locality_v3":
        gate_module = TokenLocalityGateModuleV3(
            model=model,
            block_indices=cfg.regularized_blocks,
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            grid=cfg.grid_h,
            gate_distance_scale=getattr(cfg, "gate_distance_scale", 2.0),
            gate_scale=getattr(cfg, "gate_scale", 2.0),
            device=device,
            weight_std=getattr(cfg, "gate_weight_std", 0.02),
        ).to(device)
    elif gate_type == "token_locality_v3_vec":
        gate_module = VectorKernelGateModule(
            model=model,
            block_indices=cfg.regularized_blocks,
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            grid=cfg.grid_h,
            gate_distance_scale=getattr(cfg, "gate_distance_scale", 2.0),
            gate_scale=getattr(cfg, "gate_scale", 2.0),
            device=device,
            num_basis=getattr(cfg, "gate_num_basis", 16),
            hidden_dim=getattr(cfg, "gate_hidden_dim", 64),
            weight_std=getattr(cfg, "gate_weight_std", 0.02),
        ).to(device)
    else:
        raise ValueError(f"Unsupported reg_type: {gate_type}. Use token_locality_v3 or token_locality_v3_vec.")

    if "gate_state_dict" in ckpt:
        gate_module.load_state_dict(ckpt["gate_state_dict"])
        print("  Gate weights loaded.")
    else:
        print("  WARNING: no gate_state_dict in checkpoint — using random weights.")

    # ── Data ──
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    print(f"Loading CUB test split...")
    ds = load_dataset(cfg.hf_dataset)
    test_split = ds.get("test", ds.get("validation"))

    if args.image_indices is not None:
        indices = args.image_indices
    else:
        rng = np.random.RandomState(42)
        indices = sorted(rng.choice(len(test_split), size=args.n_images,
                                    replace=False).tolist())
    print(f"  Image indices: {indices}")

    images_tensor = []
    for idx in indices:
        ex = test_split[int(idx)]
        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        images_tensor.append(val_transform(img))

    batch = torch.stack(images_tensor)   # (N, 3, 224, 224)

    # ── Extract gate fields ──
    print(f"\nExtracting gate fields ({gate_type})...")
    if gate_type == "token_locality_v3":
        fields = extract_scalar_gate_fields(model, gate_module, batch, device)
    else:
        fields = extract_vector_gate_fields(model, gate_module, batch, device)

    for b in sorted(fields.keys()):
        f = fields[b]
        print(f"  Block {b:2d}: mean={f.mean():+.4f}  std={f.std():.4f}  "
              f"range=[{f.min():+.4f}, {f.max():+.4f}]")

    # ── Plots ──
    grid_size = cfg.grid_h   # 14 for ViT-S/16 at 224px

    print("\nPlot 1/3 — per-image gate grids...")
    for img_i in range(len(indices)):
        plot_gate_grid(images_tensor[img_i], fields, cfg.regularized_blocks,
                       cfg.num_heads, grid_size, img_i, args.output_dir, gate_type)

    print("\nPlot 2/3 — mean gate field across all images...")
    plot_mean_gate_field(fields, cfg.regularized_blocks, cfg.num_heads,
                         grid_size, args.output_dir, gate_type)

    print("\nPlot 3/3 — overlays on images (active heads only)...")
    for img_i in range(len(indices)):
        plot_overlay(images_tensor[img_i], fields, cfg.regularized_blocks,
                     cfg.num_heads, grid_size, img_i, args.output_dir,
                     gate_type, threshold=args.head_threshold)

    print(f"\nDone. All outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
