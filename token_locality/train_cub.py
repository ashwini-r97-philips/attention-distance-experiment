"""Unified CUB-200-2011 training script for baseline and token-locality experiments.

Supports:
  - Baseline ViT-S/16 fine-tuning (reg_type: none)
  - Token-dependent locality (reg_type: token_locality)

Usage:
  python train_cub.py --config configs/cub/baseline_cub_vit_s16.yaml
  python train_cub.py --config configs/cub/token_locality_mild_cub_vit_s16.yaml
  python train_cub.py --config configs/cub/token_locality_medium_cub_vit_s16.yaml

  # Debug 2-epoch run:
  python train_cub.py --config configs/cub/baseline_cub_vit_s16.yaml --epochs 2 --debug
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "bimodal_head_specialisation"))
sys.path.insert(0, _HERE)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import timm
from datasets import load_dataset

from common.config import load_config, save_config
from common.model_utils import load_vit_small
from common.attention_hooks import (
    patch_attention_forward,
    unpatch_attention_forward,
    get_cached_attn_weights,
    _attention_forward_with_weights,
)
from common.mad_metrics import (
    build_distance_matrix,
    compute_mad,
    compute_local_mass,
    compute_attention_entropy,
)
from token_locality_gate_v2 import (
    TokenLocalityGateModuleV2,
    FixedLocalityPrior,
)
from token_locality_gate_v3 import TokenLocalityGateModuleV3, VectorKernelGateModule


# ─── CUB Dataset ─────────────────────────────────────────────────────────────

class CUBDataset(Dataset):
    """Map-style dataset wrapping HuggingFace CUB-200-2011."""

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


def build_train_transform(cfg):
    return transforms.Compose([
        transforms.RandomResizedCrop(
            cfg.img_size,
            scale=(0.7, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])


def build_val_transform(cfg):
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])


def get_cub_loaders(cfg, batch_size):
    """Load CUB-200-2011 train/test splits as map-style DataLoaders."""
    print(f"Loading CUB dataset: {cfg.hf_dataset}")
    ds = load_dataset(cfg.hf_dataset)

    train_split = ds.get("train")
    test_split = ds.get("test", ds.get("validation"))
    if train_split is None:
        raise ValueError(f"No 'train' split in {cfg.hf_dataset}")
    if test_split is None:
        raise ValueError(f"No 'test' or 'validation' split in {cfg.hf_dataset}")

    nw = getattr(cfg, "num_workers", 4)
    persistent = getattr(cfg, "persistent_workers", True) and nw > 0
    pf = getattr(cfg, "prefetch_factor", 4) if nw > 0 else None

    train_ds = CUBDataset(train_split, build_train_transform(cfg))
    test_ds = CUBDataset(test_split, build_val_transform(cfg))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
        persistent_workers=persistent, prefetch_factor=pf,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True, drop_last=False,
        persistent_workers=persistent, prefetch_factor=pf,
    )
    print(f"  Train: {len(train_ds)} images, {len(train_loader)} batches")
    print(f"  Test:  {len(test_ds)} images, {len(test_loader)} batches")
    return train_loader, test_loader, test_ds


# ─── Attention eval subset (deterministic) ────────────────────────────────────

def get_attention_eval_loader(test_ds, cfg, batch_size):
    """Fixed subset of test set for attention stats (no shuffle)."""
    n = min(getattr(cfg, "attention_eval_subset_size", 512), len(test_ds))
    rng = np.random.RandomState(cfg.seed)
    indices = sorted(rng.choice(len(test_ds), size=n, replace=False).tolist())
    subset = torch.utils.data.Subset(test_ds, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    return loader, indices


# ─── Schedules ────────────────────────────────────────────────────────────────

def cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(min_lr / optimizer.defaults["lr"],
                   0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── Validation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    correct1 = correct5 = total = 0
    total_loss = 0.0
    for images, targets in val_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            outputs = model(images)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * images.size(0)
        _, pred5 = outputs.topk(5, dim=1)
        correct1 += (pred5[:, 0] == targets).sum().item()
        correct5 += (pred5 == targets.unsqueeze(1)).any(dim=1).sum().item()
        total += targets.size(0)
    return {
        "val_loss": total_loss / max(1, total),
        "val_acc1": 100.0 * correct1 / max(1, total),
        "val_acc5": 100.0 * correct5 / max(1, total),
    }


# ─── Attention statistics ─────────────────────────────────────────────────────

@torch.no_grad()
def compute_attention_stats(model, loader, device, dist_matrix, cfg, gate_module=None):
    """Compute MAD, entropy, local mass for all blocks.

    If gate_module is present: gate blocks already cache attn weights.
    Non-gate blocks get temporarily patched for weight caching.
    """
    model.eval()
    all_blocks = list(range(cfg.num_blocks))
    gate_block_set = set(gate_module.block_indices) if gate_module else set()
    non_gate_blks = [b for b in all_blocks if b not in gate_block_set]

    # Temporarily patch non-gate blocks for weight caching
    originals = {}
    for b in non_gate_blks:
        attn_module = model.blocks[b].attn
        originals[b] = attn_module.forward
        attn_module.fused_attn = False
        attn_module.forward = _attention_forward_with_weights.__get__(
            attn_module, type(attn_module)
        )

    accum = {b: {"mad": [], "entropy": []} for b in all_blocks}
    for tau in cfg.tau_values:
        for b in all_blocks:
            accum[b][f"lm_{tau}"] = []

    try:
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                _ = model(images)
            attn_dict = get_cached_attn_weights(model, all_blocks)
            for b in all_blocks:
                if b not in attn_dict:
                    continue
                a = attn_dict[b]
                accum[b]["mad"].append(compute_mad(a, dist_matrix).cpu().numpy())
                accum[b]["entropy"].append(compute_attention_entropy(a).cpu().numpy())
                for tau in cfg.tau_values:
                    accum[b][f"lm_{tau}"].append(
                        compute_local_mass(a, dist_matrix, tau=tau).cpu().numpy()
                    )
    finally:
        # Restore non-gate blocks
        for b in non_gate_blks:
            attn_module = model.blocks[b].attn
            attn_module.forward = originals[b]
            attn_module.fused_attn = True

    stats = {}
    for b in all_blocks:
        stats[b] = {}
        for key in accum[b]:
            if accum[b][key]:
                stats[b][key] = np.mean(accum[b][key], axis=0).tolist()
    return stats


# ─── Gate statistics ──────────────────────────────────────────────────────────

def log_gate_statistics(gate_module, epoch, output_dir):
    """Log per-block gate statistics and save to JSON."""
    stats = gate_module.gate_statistics()
    path = os.path.join(output_dir, "gate_stats", f"epoch_{epoch:04d}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


# ─── Training loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, cfg,
                    gate_module=None, grad_accum_steps=1):
    model.train()
    total_loss = total_samples = 0.0
    total_grad_norm = 0.0
    n_steps = 0
    optimizer.zero_grad()

    for step, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets) / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            params = list(model.parameters())
            if gate_module is not None:
                params += list(gate_module.parameters())
            gn = torch.nn.utils.clip_grad_norm_(params, max_norm=cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_grad_norm += gn.item()
            n_steps += 1

        total_loss += loss.item() * grad_accum_steps * images.size(0)
        total_samples += images.size(0)

    return {
        "train_loss": total_loss / max(1, total_samples),
        "grad_norm": total_grad_norm / max(1, n_steps),
    }


# ─── Auto batch size ─────────────────────────────────────────────────────────

def find_max_batch_size(cfg, device, start=256):
    """Find largest batch size that doesn't OOM."""
    from token_locality_gate import TokenLocalityGateModule

    bs = start
    while bs >= 16:
        try:
            model = timm.create_model(cfg.model_name, pretrained=False, num_classes=cfg.num_classes)
            model = model.to(device)
            if getattr(cfg, "channels_last", False):
                model = model.to(memory_format=torch.channels_last)

            if cfg.reg_type == "token_locality":
                gate = TokenLocalityGateModule(
                    model=model,
                    block_indices=cfg.regularized_blocks,
                    embed_dim=cfg.embed_dim,
                    grid=cfg.grid_h,
                    gate_init_scale=getattr(cfg, "gate_init_scale", 0.01),
                    gate_distance_scale=getattr(cfg, "gate_distance_scale", 4.0),
                    device=device,
                )
                gate = gate.to(device)
            elif cfg.reg_type == "token_locality_v2":
                gate = TokenLocalityGateModuleV2(
                    model=model,
                    block_indices=cfg.regularized_blocks,
                    embed_dim=cfg.embed_dim,
                    grid=cfg.grid_h,
                    gate_distance_scale=getattr(cfg, "gate_distance_scale", 4.0),
                    device=device,
                    init_bias=getattr(cfg, "gate_init_bias", -5.0),
                    weight_std=getattr(cfg, "gate_weight_std", 0.02),
                )
                gate = gate.to(device)
            elif cfg.reg_type == "token_locality_v3":
                gate = TokenLocalityGateModuleV3(
                    model=model,
                    block_indices=cfg.regularized_blocks,
                    embed_dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    grid=cfg.grid_h,
                    gate_distance_scale=getattr(cfg, "gate_distance_scale", 2.0),
                    gate_scale=getattr(cfg, "gate_scale", 2.0),
                    device=device,
                    weight_std=getattr(cfg, "gate_weight_std", 0.02),
                )
                gate = gate.to(device)
            elif cfg.reg_type == "token_locality_v3_vec":
                gate = VectorKernelGateModule(
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
                )
                gate = gate.to(device)
            elif cfg.reg_type == "fixed_locality_prior":
                gate = FixedLocalityPrior(
                    model=model,
                    block_indices=cfg.regularized_blocks,
                    grid=cfg.grid_h,
                    fixed_strength=getattr(cfg, "fixed_strength", 0.5),
                    gate_distance_scale=getattr(cfg, "gate_distance_scale", 4.0),
                    device=device,
                )

            # Create optimizer to account for Adam state memory
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            scaler = torch.amp.GradScaler("cuda")

            dummy = torch.randn(bs, 3, cfg.img_size, cfg.img_size, device=device)
            if getattr(cfg, "channels_last", False):
                dummy = dummy.to(memory_format=torch.channels_last)

            # Run 2 steps to fully allocate optimizer states
            for _ in range(2):
                optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    out = model(dummy)
                    loss = out.sum()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            del model, dummy, out, loss, optimizer, scaler
            if 'gate' in dir():
                del gate
            torch.cuda.empty_cache()
            print(f"  Batch size {bs}: OK")
            return bs
        except torch.cuda.OutOfMemoryError:
            print(f"  Batch size {bs}: OOM, trying {bs // 2}")
            del model
            if 'gate' in dir():
                del gate
            torch.cuda.empty_cache()
            bs //= 2
        except Exception as e:
            print(f"  Batch size {bs}: error {e}, trying {bs // 2}")
            torch.cuda.empty_cache()
            bs //= 2

    return 16


# ─── CSV logging helpers ──────────────────────────────────────────────────────

def append_csv(path, row_dict):
    """Append a row to a CSV file, creating headers if needed."""
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row_dict)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CUB-200 training (baseline + token locality)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--debug", action="store_true", help="Debug mode: extra prints, no compile")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    device = torch.device(cfg.device)

    output_dir = cfg.output_dir
    for subdir in ["checkpoints", "attention_stats", "gate_stats"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    save_config(cfg, os.path.join(output_dir, "config.yaml"))

    # Git commit
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = "unknown"
    with open(os.path.join(output_dir, "git_commit.txt"), "w") as f:
        f.write(commit + "\n")

    # Seeds & CUDA settings
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if getattr(cfg, "cudnn_benchmark", True):
        torch.backends.cudnn.benchmark = True

    # GPU info
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print("=" * 70)
    print(f"CUB-200 TRAINING — {os.path.basename(output_dir)}")
    print(f"  GPU:            {gpu_name}")
    print(f"  reg_type:       {cfg.reg_type}")
    print(f"  epochs:         {cfg.epochs}")
    print(f"  lr:             {cfg.lr}")
    if cfg.reg_type in ("token_locality", "token_locality_v2", "fixed_locality_prior"):
        print(f"  gate_blocks:    {cfg.regularized_blocks}")
        print(f"  gate_scale:     {getattr(cfg, 'gate_distance_scale', 4.0)}")
        if cfg.reg_type == "fixed_locality_prior":
            print(f"  fixed_strength: {getattr(cfg, 'fixed_strength', 0.5)}")
        if cfg.reg_type == "token_locality_v2":
            print(f"  gate_init_bias: {getattr(cfg, 'gate_init_bias', -5.0)}")
            print(f"  gate_lr_mult:   {getattr(cfg, 'gate_lr_multiplier', 10.0)}")
    print("=" * 70)

    # ── Auto batch size ──
    requested_bs = cfg.batch_size
    if requested_bs <= 0 or requested_bs > 1024:
        requested_bs = 512
    print(f"\nFinding max batch size (starting at {requested_bs})...")
    batch_size = find_max_batch_size(cfg, device, start=requested_bs)
    print(f"  Using batch size: {batch_size}")
    grad_accum = max(1, 512 // batch_size)
    print(f"  Gradient accumulation: {grad_accum} (effective: {batch_size * grad_accum})")

    # ── Model ──
    print("\nLoading pretrained ViT-S/16...")
    model = load_vit_small(cfg, pretrained=True)
    if getattr(cfg, "channels_last", False):
        model = model.to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # ── Token locality gate / fixed prior ──
    gate_module = None
    if cfg.reg_type == "token_locality":
        from token_locality_gate import TokenLocalityGateModule
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
        n_gate = sum(p.numel() for p in gate_module.parameters())
        print(f"  Gate module (v1) installed on blocks {cfg.regularized_blocks}")
        print(f"  Gate parameters: {n_gate:,}")
    elif cfg.reg_type == "token_locality_v2":
        gate_module = TokenLocalityGateModuleV2(
            model=model,
            block_indices=cfg.regularized_blocks,
            embed_dim=cfg.embed_dim,
            grid=cfg.grid_h,
            gate_distance_scale=getattr(cfg, "gate_distance_scale", 4.0),
            device=device,
            init_bias=getattr(cfg, "gate_init_bias", -5.0),
            weight_std=getattr(cfg, "gate_weight_std", 0.02),
        )
        gate_module = gate_module.to(device)
        n_gate = sum(p.numel() for p in gate_module.parameters())
        print(f"  Gate module (v2) installed on blocks {cfg.regularized_blocks}")
        print(f"  Gate parameters: {n_gate:,}")
        print(f"  Gate init bias: {getattr(cfg, 'gate_init_bias', -5.0)} → Softplus output ≈ {F.softplus(torch.tensor(getattr(cfg, 'gate_init_bias', -5.0))).item():.5f}")
    elif cfg.reg_type == "token_locality_v3":
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
        )
        gate_module = gate_module.to(device)
        n_gate = sum(p.numel() for p in gate_module.parameters())
        gate_scale = getattr(cfg, "gate_scale", 2.0)
        print(f"  Gate module (v3) installed on blocks {cfg.regularized_blocks}")
        print(f"  Gate parameters: {n_gate:,}  (per-head, signed tanh × {gate_scale})")
    elif cfg.reg_type == "token_locality_v3_vec":
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
        )
        gate_module = gate_module.to(device)
        n_gate = sum(p.numel() for p in gate_module.parameters())
        num_basis = getattr(cfg, "gate_num_basis", 16)
        hidden_dim = getattr(cfg, "gate_hidden_dim", 64)
        print(f"  Gate module (v3-vec) installed on blocks {cfg.regularized_blocks}")
        print(f"  Gate parameters: {n_gate:,}  (MLP→{hidden_dim}→H×{num_basis} RBF basis, signed tanh)")
    elif cfg.reg_type == "fixed_locality_prior":
        gate_module = FixedLocalityPrior(
            model=model,
            block_indices=cfg.regularized_blocks,
            grid=cfg.grid_h,
            fixed_strength=getattr(cfg, "fixed_strength", 0.5),
            gate_distance_scale=getattr(cfg, "gate_distance_scale", 4.0),
            device=device,
        )
        print(f"  Fixed locality prior on blocks {cfg.regularized_blocks}")
        print(f"  fixed_strength={getattr(cfg, 'fixed_strength', 0.5)} × gate_distance_scale={getattr(cfg, 'gate_distance_scale', 4.0)}")

    # ── Data ──
    train_loader, test_loader, test_ds = get_cub_loaders(cfg, batch_size)
    attn_eval_loader, attn_eval_indices = get_attention_eval_loader(test_ds, cfg, batch_size)

    # ── Optimizer + Scheduler ──
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    param_groups = [{"params": list(model.parameters()), "lr": cfg.lr}]
    if gate_module is not None and hasattr(gate_module, "parameters"):
        gate_params = list(gate_module.parameters())
        if gate_params:  # v1 and v2 have learnable params; fixed prior has none
            gate_lr = cfg.lr * getattr(cfg, "gate_lr_multiplier", 1.0)
            param_groups.append({"params": gate_params, "lr": gate_lr})
            if getattr(cfg, "gate_lr_multiplier", 1.0) != 1.0:
                print(f"  Gate LR: {gate_lr:.2e} ({getattr(cfg, 'gate_lr_multiplier', 1.0)}× base)")
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    scheduler = cosine_schedule_with_warmup(optimizer, cfg.warmup_epochs, cfg.epochs)
    scaler = torch.amp.GradScaler("cuda")

    # Distance matrix for MAD
    dist_matrix = build_distance_matrix(cfg.grid_h, cfg.grid_w, device=device)

    # ── Resume ──
    start_epoch = 1
    best_acc1 = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if gate_module and "gate_state_dict" in ckpt:
            gate_module.load_state_dict(ckpt["gate_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc1 = ckpt.get("val_acc1", 0.0)
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f"  Resumed from epoch {start_epoch}, best_acc1={best_acc1:.2f}%")

    # ── Debug: verify gradients on first batch ──
    if args.debug and cfg.reg_type in ("token_locality", "token_locality_v2"):
        print("\n[DEBUG] Verifying gate gradients on one batch...")
        model.train()
        imgs, tgts = next(iter(train_loader))
        imgs = imgs.to(device, non_blocking=True)
        tgts = tgts.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            out = model(imgs)
            loss = criterion(out, tgts)
        scaler.scale(loss).backward()
        for i, gate in enumerate(gate_module.gates):
            g = gate.linear.weight.grad
            if g is not None:
                print(f"    Block {cfg.regularized_blocks[i]} gate grad: "
                      f"norm={g.norm().item():.6f}, mean={g.mean().item():.8f}")
            else:
                print(f"    Block {cfg.regularized_blocks[i]} gate grad: None ⚠")
        optimizer.zero_grad()
        print("  [DEBUG] Gate gradient verification complete.\n")

    # ── Training loop ─────────────────────────────────────────────────────
    peak_mem_gb = 0.0
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch}/{cfg.epochs}  lr={scheduler.get_last_lr()[0]:.2e}")

        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            cfg=cfg,
            gate_module=gate_module,
            grad_accum_steps=grad_accum,
        )
        scheduler.step()

        val_metrics = validate(model, test_loader, criterion, device)

        # Track peak GPU memory
        if device.type == "cuda":
            mem = torch.cuda.max_memory_allocated(device) / 1e9
            peak_mem_gb = max(peak_mem_gb, mem)

        elapsed = time.time() - t0
        imgs_per_sec = len(train_loader.dataset) / elapsed

        print(f"  train_loss={train_metrics['train_loss']:.4f}  "
              f"val_acc1={val_metrics['val_acc1']:.2f}%  "
              f"val_acc5={val_metrics['val_acc5']:.2f}%  "
              f"time={elapsed:.0f}s  {imgs_per_sec:.0f} img/s")

        # Attention stats
        attn_stats = compute_attention_stats(
            model, attn_eval_loader, device, dist_matrix, cfg, gate_module
        )
        attn_path = os.path.join(output_dir, "attention_stats", f"epoch_{epoch:04d}.json")
        with open(attn_path, "w") as f:
            json.dump(attn_stats, f)

        # Gate stats
        gate_stats_dict = None
        if gate_module is not None:
            gate_stats_dict = log_gate_statistics(gate_module, epoch, output_dir)
            print("  [Gate] ", end="")
            for bk, s in gate_stats_dict.items():
                if "gate_output_mean" in s:
                    per_head = s.get("per_head_mean", [])
                    head_str = ("[" + " ".join(f"{v:+.2f}" for v in per_head) + "]") if per_head else ""
                    print(f"{bk}: out={s['gate_output_mean']:.4f}±{s.get('gate_output_std', 0):.4f} {head_str}  ", end="")
                else:
                    print(f"{bk}: μ={s['weight_mean']:+.4f} σ={s['weight_std']:.4f}  ", end="")
            print()

        # CSV logging
        log_row = {
            "epoch": epoch,
            "train_loss": round(train_metrics["train_loss"], 6),
            "val_loss": round(val_metrics["val_loss"], 6),
            "val_acc1": round(val_metrics["val_acc1"], 4),
            "val_acc5": round(val_metrics["val_acc5"], 4),
            "grad_norm": round(train_metrics["grad_norm"], 4),
            "lr": scheduler.get_last_lr()[0],
            "elapsed_s": round(elapsed, 1),
            "imgs_per_sec": round(imgs_per_sec, 0),
            "peak_mem_gb": round(peak_mem_gb, 2),
        }
        append_csv(os.path.join(output_dir, "training_log.csv"), log_row)

        # MAD stats CSV (per-layer, per-head)
        for b in range(cfg.num_blocks):
            if b in attn_stats and "mad" in attn_stats[b]:
                for h_idx, mad_val in enumerate(attn_stats[b]["mad"]):
                    mad_row = {
                        "epoch": epoch,
                        "block": b,
                        "head": h_idx,
                        "mad": round(float(mad_val), 6),
                    }
                    if "entropy" in attn_stats[b]:
                        mad_row["entropy"] = round(float(attn_stats[b]["entropy"][h_idx]), 6)
                    for tau in cfg.tau_values:
                        key = f"lm_{tau}"
                        if key in attn_stats[b]:
                            mad_row[f"local_mass_{tau}"] = round(
                                float(attn_stats[b][key][h_idx]), 6
                            )
                    append_csv(os.path.join(output_dir, "per_layer_head_mad.csv"), mad_row)

        # Gate stats CSV
        if gate_stats_dict:
            for bk, s in gate_stats_dict.items():
                gate_row = {
                    "epoch": epoch,
                    "block": bk,
                    "weight_mean": round(s.get("weight_mean", 0), 6),
                    "weight_std": round(s.get("weight_std", 0), 6),
                    "weight_min": round(s.get("weight_min", 0), 6),
                    "weight_max": round(s.get("weight_max", 0), 6),
                    "bias_mean": round(s.get("bias_mean", s.get("bias", 0)), 6),
                    "bias_std": round(s.get("bias_std", 0), 6),
                    "gate_output_mean": round(s.get("gate_output_mean", 0), 6),
                    "gate_output_std": round(s.get("gate_output_std", 0), 6),
                }
                append_csv(os.path.join(output_dir, "gate_stats.csv"), gate_row)

        # Checkpoint
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc1": val_metrics["val_acc1"],
            "val_acc5": val_metrics["val_acc5"],
        }
        if gate_module is not None:
            ckpt_data["gate_state_dict"] = gate_module.state_dict()

        # Save last checkpoint (overwrite each epoch)
        torch.save(ckpt_data, os.path.join(output_dir, "checkpoints", "last.pth"))

        # Best by val_acc1
        if val_metrics["val_acc1"] > best_acc1:
            best_acc1 = val_metrics["val_acc1"]
            torch.save(ckpt_data, os.path.join(output_dir, "checkpoints", "best.pth"))
            print(f"  ★ New best: {best_acc1:.2f}%")

    # ── Runtime summary ───────────────────────────────────────────────────
    summary_lines = [
        f"Experiment: {os.path.basename(output_dir)}",
        f"GPU: {gpu_name}",
        f"Peak memory: {peak_mem_gb:.2f} GB",
        f"Batch size: {batch_size}",
        f"Gradient accumulation: {grad_accum}",
        f"Best val_acc1: {best_acc1:.2f}%",
        f"Total epochs: {cfg.epochs}",
        f"reg_type: {cfg.reg_type}",
    ]
    if cfg.reg_type == "token_locality":
        summary_lines.append(f"gate_blocks: {cfg.regularized_blocks}")
        summary_lines.append(f"gate_distance_scale: {getattr(cfg, 'gate_distance_scale', 4.0)}")

    with open(os.path.join(output_dir, "runtime_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE — {os.path.basename(output_dir)}")
    print(f"  Best val_acc1: {best_acc1:.2f}%")
    print(f"  Peak GPU mem:  {peak_mem_gb:.2f} GB")
    print(f"  Results →      {os.path.abspath(output_dir)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
