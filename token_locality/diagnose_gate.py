"""Diagnose why the token locality gate isn't learning.

Checks:
  1. Gate OUTPUT magnitudes (not just weights) — Softplus(near-zero) ≈ ln(2)
  2. Effective penalty magnitude vs attention logit magnitude
  3. Gradient flow through the gate
  4. Whether distance penalty dominates the gate contribution
  5. Gate learning rate sufficiency

Usage:
  python diagnose_gate.py --checkpoint runs/cub/token_locality_mild_cub_vit_s16/checkpoints/best.pth
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "bimodal_head_specialisation"))
sys.path.insert(0, _HERE)

import torch
import torch.nn.functional as F
import numpy as np
import timm
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from common.config import load_config
from token_locality_gate import (
    TokenLocalityGateModule,
    build_patch_distance_matrix,
    LocalityGateBranch,
)
from common.mad_metrics import build_distance_matrix


class SimpleCUBDataset(Dataset):
    def __init__(self, hf_split, transform):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img), ex["label"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Infer config from checkpoint path
    if args.config is None:
        ckpt_dir = os.path.dirname(os.path.dirname(args.checkpoint))
        config_path = os.path.join(ckpt_dir, "config.yaml")
        if not os.path.exists(config_path):
            # Try to find it
            run_name = os.path.basename(ckpt_dir)
            config_path = f"configs/cub/{run_name.replace('_cub_vit_s16', '')}_cub_vit_s16.yaml"
        args.config = config_path

    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    cfg = load_config(args.config)

    # Load model
    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=200)
    model = model.to(device)

    # Install gate
    gate_module = TokenLocalityGateModule(
        model=model,
        block_indices=cfg.regularized_blocks,
        embed_dim=cfg.embed_dim,
        grid=cfg.grid_h,
        gate_init_scale=getattr(cfg, "gate_init_scale", 0.01),
        gate_distance_scale=getattr(cfg, "gate_distance_scale", 4.0),
        device=device,
    )
    gate_module = gate_module.to(device)

    # Load weights
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if "gate_state_dict" in ckpt:
        gate_module.load_state_dict(ckpt["gate_state_dict"])
        print("[OK] Gate state loaded from checkpoint")
    else:
        print("[WARN] No gate state in checkpoint!")

    # Load a small batch of data
    ds = load_dataset("bentrevett/caltech-ucsd-birds-200-2011", split="test")
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_ds = SimpleCUBDataset(ds, transform)
    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    imgs, labels = next(iter(loader))
    imgs = imgs.to(device)
    labels = labels.to(device)

    print(f"\n{'='*70}")
    print("GATE DIAGNOSTIC REPORT")
    print(f"{'='*70}")

    # ─── 1. Gate weight statistics ───────────────────────────────────────
    print("\n┌── 1. Gate Weight Statistics ──┐")
    for i, block_idx in enumerate(cfg.regularized_blocks):
        gate = gate_module.gates[i]
        w = gate.linear.weight.detach()
        b = gate.linear.bias.detach()
        print(f"  Block {block_idx}: weight mean={w.mean():.6f} std={w.std():.6f} "
              f"min={w.min():.6f} max={w.max():.6f} bias={b.item():.6f}")

    # ─── 2. Gate OUTPUT magnitudes ───────────────────────────────────────
    print("\n┌── 2. Gate Output Magnitudes (on real data) ──┐")
    print("  (This is what matters — Softplus(near-zero-input) ≈ ln(2) ≈ 0.693)")
    model.eval()

    # Hook to capture intermediate representations at each gated block
    block_norms = {}
    def make_hook(block_idx):
        def hook_fn(module, input, output):
            # input[0] is x (already normed by block.norm1 before attn call)
            block_norms[block_idx] = input[0].detach()
        return hook_fn

    hooks = []
    for block_idx in cfg.regularized_blocks:
        h = model.blocks[block_idx].attn.register_forward_hook(make_hook(block_idx))
        hooks.append(h)

    with torch.no_grad():
        _ = model(imgs)

    for h in hooks:
        h.remove()

    gate_outputs_by_block = {}
    for i, block_idx in enumerate(cfg.regularized_blocks):
        x_norm = block_norms[block_idx]  # (B, N, D) — normed input to attn
        gate = gate_module.gates[i]
        gate_out = gate(x_norm)  # (B, N, 1)
        gate_outputs_by_block[block_idx] = gate_out.squeeze(-1)  # (B, N)

        # Statistics
        g_flat = gate_out.squeeze(-1)[:, 1:]  # patch tokens only
        print(f"  Block {block_idx}: gate_output mean={g_flat.mean():.6f} "
              f"std={g_flat.std():.6f} min={g_flat.min():.6f} max={g_flat.max():.6f}")

    # ─── 3. What Softplus(input) evaluates to ───────────────────────────
    print("\n┌── 3. Pre-activation values (Linear output before Softplus) ──┐")
    for i, block_idx in enumerate(cfg.regularized_blocks):
        x_norm = block_norms[block_idx]
        gate = gate_module.gates[i]
        pre_act = gate.linear(x_norm)[:, 1:, 0]  # (B, Np) before softplus
        print(f"  Block {block_idx}: pre_activation mean={pre_act.mean():.6f} "
              f"std={pre_act.std():.6f} min={pre_act.min():.6f} max={pre_act.max():.6f}")
        # Softplus(0) = ln(2) ≈ 0.693
        print(f"    → Softplus maps these to ≈ {F.softplus(pre_act.mean()):.4f}")

    # ─── 4. Effective penalty magnitude vs attention logits ──────────────
    print("\n┌── 4. Penalty vs Attention Logit Scale ──┐")
    dist_mat = gate_module.dist_matrix
    dist_scale = getattr(cfg, "gate_distance_scale", 4.0)
    max_dist = dist_mat.max().item()
    mean_dist = dist_mat[dist_mat > 0].mean().item()
    print(f"  Distance matrix: max={max_dist:.4f} mean={mean_dist:.4f}")
    print(f"  gate_distance_scale: {dist_scale}")

    for i, block_idx in enumerate(cfg.regularized_blocks[:3]):
        g_mean = gate_outputs_by_block[block_idx][:, 1:].mean().item()
        penalty_mean = g_mean * mean_dist * dist_scale
        penalty_max = g_mean * max_dist * dist_scale
        print(f"  Block {block_idx}: avg_penalty={penalty_mean:.4f} "
              f"max_penalty={penalty_max:.4f} "
              f"(gate_out={g_mean:.4f} × dist={mean_dist:.4f} × scale={dist_scale})")

    # Now capture actual attention logits to compare
    print("\n  Comparing to actual attention logit magnitudes:")
    attn_logit_stats = {}
    def make_logit_hook(block_idx):
        def hook_fn(module, input, output):
            x = input[0]
            B, N, C = x.shape
            qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = module.q_norm(q), module.k_norm(k)
            attn = (q * module.scale) @ k.transpose(-2, -1)
            attn_logit_stats[block_idx] = {
                "mean": attn[:, :, 1:, 1:].mean().item(),
                "std": attn[:, :, 1:, 1:].std().item(),
                "absmax": attn[:, :, 1:, 1:].abs().max().item(),
            }
        return hook_fn

    # Need to temporarily remove gate forwards to get raw logits
    gate_module.remove_gates(model)
    hooks = []
    for block_idx in cfg.regularized_blocks[:3]:
        h = model.blocks[block_idx].attn.register_forward_hook(make_logit_hook(block_idx))
        hooks.append(h)
    with torch.no_grad():
        _ = model(imgs)
    for h in hooks:
        h.remove()

    for block_idx in cfg.regularized_blocks[:3]:
        s = attn_logit_stats[block_idx]
        g_mean = gate_outputs_by_block[block_idx][:, 1:].mean().item()
        penalty_mean = g_mean * mean_dist * dist_scale
        ratio = penalty_mean / (s["std"] + 1e-8)
        print(f"    Block {block_idx}: logit mean={s['mean']:.3f} std={s['std']:.3f} "
              f"absmax={s['absmax']:.3f}  → penalty/logit_std = {ratio:.4f}")

    # ─── 5. Gradient analysis with fresh forward ─────────────────────────
    print("\n┌── 5. Gradient Analysis ──┐")
    # Reinstall gates
    gate_module2 = TokenLocalityGateModule(
        model=model,
        block_indices=cfg.regularized_blocks,
        embed_dim=cfg.embed_dim,
        grid=cfg.grid_h,
        gate_init_scale=getattr(cfg, "gate_init_scale", 0.01),
        gate_distance_scale=getattr(cfg, "gate_distance_scale", 4.0),
        device=device,
    )
    gate_module2 = gate_module2.to(device)
    if "gate_state_dict" in ckpt:
        gate_module2.load_state_dict(ckpt["gate_state_dict"])

    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    logits = model(imgs)
    loss = criterion(logits, labels)
    loss.backward()

    print("  Per-block gradient stats:")
    for i, block_idx in enumerate(cfg.regularized_blocks):
        gate = gate_module2.gates[i]
        w_grad = gate.linear.weight.grad
        b_grad = gate.linear.bias.grad
        w = gate.linear.weight.detach()

        if w_grad is not None:
            # Compute update magnitude relative to weight magnitude
            w_norm = w.norm().item()
            g_norm = w_grad.norm().item()
            # With lr=1e-5, AdamW would do approximately:
            lr = cfg.lr
            update_est = lr * g_norm / (w_norm + 1e-8)
            print(f"    Block {block_idx}: |w|={w_norm:.6f} |grad|={g_norm:.4f} "
                  f"|bias_grad|={b_grad.item():.6f} "
                  f"estimated_relative_update={update_est:.6f}")
        else:
            print(f"    Block {block_idx}: NO gradient!")

    # ─── 6. Root cause diagnosis ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("DIAGNOSIS")
    print(f"{'='*70}")

    g0_out = gate_outputs_by_block[cfg.regularized_blocks[0]][:, 1:].mean().item()
    pre_act_mean = 0.0  # near-zero input to Softplus
    softplus_at_zero = float(F.softplus(torch.tensor(0.0)))

    print(f"""
  1. Softplus(near-zero) ≈ {softplus_at_zero:.4f} (= ln(2))
     → Even with zero-init weights, gate OUTPUT is ~0.69, NOT ~0.0!
     → The gate produces a CONSTANT ~0.69 regardless of input.
     → Actual gate output: {g0_out:.4f}

  2. The effective penalty is: {g0_out:.4f} × distance × {dist_scale}
     → For avg distance {mean_dist:.3f}: penalty ≈ {g0_out * mean_dist * dist_scale:.3f}
     → For max distance {max_dist:.3f}: penalty ≈ {g0_out * max_dist * dist_scale:.3f}
     → This is FIXED regardless of token content!

  3. Gate weight std ≈ {gate_module2.gates[0].linear.weight.std():.4f}
     → Linear(x) produces output with std ≈ embed_dim^0.5 × weight_std ≈ {384**0.5 * gate_module2.gates[0].linear.weight.std().item():.3f}
     → This is tiny compared to Softplus's constant offset of 0.693
     → The token-dependent VARIATION is negligible vs the constant term

  4. Core problem: The gate architecture cannot express "no penalty"
     → Softplus is bounded below by 0, but Softplus(0) = 0.693
     → To get gate ≈ 0 (no penalty), need pre-activation << 0
     → But weights are initialized near 0, so pre-activation ≈ 0
     → The model starts with a FIXED penalty of 0.693 × dist × scale

  FIXES NEEDED:
    A. Initialize bias to -5.0 so Softplus(-5) ≈ 0.007 (starts as near-baseline)
    B. Use higher LR for gate parameters (10-100× the base LR)
    C. Use a gate activation that equals 0 at initialization
       Options: ReLU(x - threshold), sigmoid(x) - 0.5, or just x²
""")


if __name__ == "__main__":
    main()
