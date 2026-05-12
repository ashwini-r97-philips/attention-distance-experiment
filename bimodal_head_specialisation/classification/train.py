"""Training script for ViT-S/16 on ImageNet-1K.

Modes (controlled by config YAML):
  reg_type: none     -> baseline CE-only finetuning
  reg_type: spread   -> CE + spread loss (MAD variance maximisation)
  reg_type: bimodal  -> CE + bimodal mixture prior on MAD

Usage:
  python train.py --config configs/baseline.yaml
  python train.py --config configs/spread_weak.yaml
"""

import argparse
import json
import math
import os
import sys
import time
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

import numpy as np
import torch
import torch.nn as nn
from timm.data.mixup import Mixup
from tqdm import tqdm

# Silence harmless PIL warnings from ImageNet JPEGs with corrupt EXIF
warnings.filterwarnings("ignore", message="Corrupt EXIF data", category=UserWarning)

from common.config import load_config, save_config
from common.model_utils import load_vit_small
from common.attention_hooks import (
    patch_attention_forward,
    get_cached_attn_weights,
    clear_cached_attn_weights,
    capture_attention,
)
from common.regularisers import build_regulariser
from common.mad_metrics import (
    build_distance_matrix,
    compute_mad,
    compute_local_mass,
    compute_attention_entropy,
    compute_inter_head_mad_variance,
)
from data import get_train_loader, get_val_loader, get_attention_eval_subset


# ─── Schedules ───────────────────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(min_lr / optimizer.defaults["lr"],
                   0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_reg_warmup_factor(epoch, warmup_epochs):
    if epoch >= warmup_epochs:
        return 1.0
    return epoch / max(1, warmup_epochs)


# ─── Validation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    correct1 = correct5 = total = 0
    total_loss = 0.0
    for images, targets in val_loader:
        images, targets = images.to(device), targets.to(device)
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


# ─── Attention stats ────────────────────────────────────────────────────────

@torch.no_grad()
def compute_attention_stats(model, loader, device, dist_matrix, cfg):
    """Compute per-layer/per-head MAD, local mass, entropy on a val subset."""
    model.eval()
    all_blocks = list(range(cfg.num_blocks))
    accum = {b: {"mad": [], "entropy": []} for b in all_blocks}
    for tau in cfg.tau_values:
        for b in all_blocks:
            accum[b][f"lm_{tau}"] = []

    for images, _ in loader:
        images = images.to(device)
        with capture_attention(model, all_blocks) as get_attn:
            _ = model(images)
            attn_dict = get_attn()
        for b in all_blocks:
            a = attn_dict[b]
            accum[b]["mad"].append(compute_mad(a, dist_matrix).cpu().numpy())
            accum[b]["entropy"].append(compute_attention_entropy(a).cpu().numpy())
            for tau in cfg.tau_values:
                accum[b][f"lm_{tau}"].append(
                    compute_local_mass(a, dist_matrix, tau=tau).cpu().numpy()
                )

    stats = {}
    for b in all_blocks:
        stats[b] = {}
        for key in accum[b]:
            stats[b][key] = np.mean(accum[b][key], axis=0).tolist()
    return stats


# ─── Training loop ───────────────────────────────────────────────────────────

def train_one_epoch(model, train_loader, optimizer, criterion, device,
                    mixup_fn, reg_fn, reg_blocks, warmup_factor, scaler, cfg):
    model.train()
    total_task = total_reg = total_samples = 0.0
    total_grad_norm = 0.0
    n_steps = 0

    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            outputs = model(images)
            task_loss = criterion(outputs, targets)

        reg_loss = torch.tensor(0.0, device=device)
        reg_info = {}
        if reg_fn is not None:
            attn_dict = get_cached_attn_weights(model, reg_blocks)
            if attn_dict:
                reg_loss, reg_info = reg_fn(attn_dict, warmup_factor=warmup_factor)

        loss = task_loss + cfg.lambda_reg * reg_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        clear_cached_attn_weights(model, reg_blocks)

        total_task += task_loss.item() * images.size(0)
        total_reg += reg_loss.item() * images.size(0)
        total_samples += images.size(0)
        total_grad_norm += gn.item()
        n_steps += 1

    return {
        "train_loss": total_task / max(1, total_samples),
        "reg_loss": total_reg / max(1, total_samples),
        "grad_norm": total_grad_norm / max(1, n_steps),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = cfg.device

    # Output directory
    output_dir = cfg.output_dir
    assert output_dir, "output_dir must be set in config"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "attention_stats"), exist_ok=True)

    # Save config
    save_config(cfg, os.path.join(output_dir, "config.yaml"))

    # Log git commit
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = "unknown"
    with open(os.path.join(output_dir, "git_commit.txt"), "w") as f:
        f.write(commit + "\n")

    # Seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    print("=" * 60)
    print(f"ViT-S/16 ImageNet-1K Training")
    print(f"  reg_type   : {cfg.reg_type}")
    print(f"  lambda_reg : {cfg.lambda_reg}")
    print(f"  epochs     : {cfg.epochs}")
    print(f"  batch_size : {cfg.batch_size}")
    print(f"  lr         : {cfg.lr}")
    print(f"  seed       : {cfg.seed}")
    print(f"  output     : {output_dir}")
    print(f"  git commit : {commit}")
    print("=" * 60)

    # Model
    print("Loading pretrained ViT-S/16...")
    model = load_vit_small(cfg, pretrained=True)

    # Regulariser
    reg_blocks = cfg.regularized_blocks
    reg_fn = build_regulariser(cfg)
    if reg_fn is not None:
        patch_attention_forward(model, reg_blocks, differentiable=True)
        print(f"  Patched blocks {reg_blocks} for differentiable attention capture")
        print(f"  Regulariser: {cfg.reg_type}, lambda={cfg.lambda_reg}")

    # Data
    print(f"Loading ImageNet from HuggingFace: {cfg.hf_dataset} (streaming)...")
    train_loader = get_train_loader(cfg)
    val_loader = get_val_loader(cfg, batch_size=max(cfg.batch_size, 256))
    attn_eval_loader, attn_eval_indices = get_attention_eval_subset(cfg)
    with open(os.path.join(output_dir, "attn_eval_indices.json"), "w") as f:
        json.dump(attn_eval_indices, f)

    # Mixup / CutMix (disabled when both alphas are 0)
    mixup_fn = None
    if cfg.mixup_alpha > 0 or cfg.cutmix_alpha > 0:
        mixup_fn = Mixup(
            mixup_alpha=cfg.mixup_alpha,
            cutmix_alpha=cfg.cutmix_alpha,
            prob=cfg.mixup_prob,
            switch_prob=cfg.mixup_switch_prob,
            num_classes=cfg.num_classes,
        )

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_epochs, cfg.epochs)
    scaler = torch.amp.GradScaler("cuda")

    # Distance matrix
    dist_matrix = build_distance_matrix(cfg.grid_h, cfg.grid_w, device=device)

    # Resume
    start_epoch = 1
    best_acc1 = 0.0
    epoch_logs = []
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc1 = ckpt.get("val_acc1", 0.0)
        for _ in range(start_epoch - 1):
            scheduler.step()
        log_path = os.path.join(output_dir, "training_log.json")
        if os.path.exists(log_path):
            with open(log_path) as f:
                epoch_logs = json.load(f)
        print(f"  Resumed at epoch {start_epoch}, best acc1={best_acc1:.2f}%")

    # ─── Training loop ───────────────────────────────────────────────────
    epoch_pbar = tqdm(range(start_epoch, cfg.epochs + 1), desc="Epochs", unit="ep")
    for epoch in epoch_pbar:
        t0 = time.time()
        wf = get_reg_warmup_factor(epoch, cfg.lambda_warmup_epochs)

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            mixup_fn, reg_fn, reg_blocks, wf, scaler, cfg,
        )
        scheduler.step()

        val_metrics = validate(model, val_loader, criterion, device)

        # Attention stats
        attn_stats = {}
        if epoch % cfg.attention_eval_frequency_epochs == 0 or epoch == cfg.epochs:
            attn_stats = compute_attention_stats(
                model, attn_eval_loader, device, dist_matrix, cfg
            )
            stats_path = os.path.join(output_dir, "attention_stats", f"epoch_{epoch:04d}.json")
            with open(stats_path, "w") as f:
                json.dump(attn_stats, f, indent=2)

        elapsed = time.time() - t0
        gpu_mem = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0

        # Extract epoch-level MAD for trajectory logging
        epoch_mads = {}
        if attn_stats:
            epoch_mads = {int(b): s["mad"] for b, s in attn_stats.items()}

        log_entry = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "reg_warmup_factor": wf,
            "time_s": elapsed,
            "gpu_mem_gb": round(gpu_mem, 2),
            "mads": epoch_mads,
        }
        epoch_logs.append(log_entry)
        with open(os.path.join(output_dir, "training_log.json"), "w") as f:
            json.dump(epoch_logs, f, indent=2)

        # Checkpoint
        is_best = val_metrics["val_acc1"] > best_acc1
        if is_best:
            best_acc1 = val_metrics["val_acc1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc1": best_acc1,
                "val_acc5": val_metrics["val_acc5"],
            }, os.path.join(output_dir, "checkpoints", "best.pth"))

        # Save every 10 epochs
        if epoch % 10 == 0 or epoch == cfg.epochs:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc1": val_metrics["val_acc1"],
            }, os.path.join(output_dir, "checkpoints", f"epoch_{epoch:04d}.pth"))

        reg_str = f" reg={train_metrics['reg_loss']:.6f}" if cfg.reg_type != "none" else ""
        best_str = " *BEST*" if is_best else ""
        epoch_pbar.set_postfix_str(
            f"loss={train_metrics['train_loss']:.4f} "
            f"acc1={val_metrics['val_acc1']:.2f}% "
            f"lr={optimizer.param_groups[0]['lr']:.2e}{best_str}"
        )
        print(f"Epoch {epoch:3d}/{cfg.epochs} | "
              f"loss={train_metrics['train_loss']:.4f}{reg_str} | "
              f"val_acc1={val_metrics['val_acc1']:.2f}% "
              f"val_acc5={val_metrics['val_acc5']:.2f}% | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} "
              f"gnorm={train_metrics['grad_norm']:.2f} | "
              f"{elapsed:.0f}s{best_str}")

    print(f"\nTraining complete. Best val_acc1: {best_acc1:.2f}%")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
