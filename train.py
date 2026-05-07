"""Training script for bimodal head specialization experiment.

Two modes:
  --mode baseline     : finetune DeiT-S without regularizer (matched training)
  --mode regularized  : finetune DeiT-S with bimodal head distance regularizer

Usage:
  CUDA_VISIBLE_DEVICES=2 python train.py --mode baseline --epochs 30
  CUDA_VISIBLE_DEVICES=2 python train.py --mode regularized --epochs 30
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from timm.data.mixup import Mixup
from tqdm import tqdm

import config as cfg
from model_utils import load_deit_small
from attention_hooks import (
    patch_attention_forward,
    get_cached_attn_weights,
    clear_cached_attn_weights,
)
from bimodal_loss import BimodalHeadLoss
from mad_metrics import build_distance_matrix, compute_mad
from data import get_train_loader, get_val_loader, get_debug_loaders


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    """Cosine LR schedule with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(min_lr / optimizer.defaults["lr"], 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_warmup_factor(epoch, warmup_epochs):
    """Linear warmup for regularizer weight."""
    if epoch >= warmup_epochs:
        return 1.0
    return epoch / max(1, warmup_epochs)


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    correct1 = 0
    correct5 = 0
    total = 0

    for images, targets in val_loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        _, pred5 = outputs.topk(5, dim=1)
        correct1 += (pred5[:, 0] == targets).sum().item()
        correct5 += (pred5 == targets.unsqueeze(1)).any(dim=1).sum().item()
        total += targets.size(0)

    acc1 = 100.0 * correct1 / total
    acc5 = 100.0 * correct5 / total
    return acc1, acc5


@torch.no_grad()
def compute_epoch_mads(model, val_loader, device, dist_matrix, num_batches=10):
    """Compute per-block MAD on a few val batches for logging."""
    from attention_hooks import capture_attention
    model.eval()
    all_blocks = list(range(cfg.NUM_BLOCKS))
    mad_accum = {b: [] for b in all_blocks}

    for batch_idx, (images, _) in enumerate(val_loader):
        if batch_idx >= num_batches:
            break
        images = images.to(device)
        with capture_attention(model, all_blocks) as get_attn:
            _ = model(images)
            attn_dict = get_attn()
        for bidx in all_blocks:
            mad_accum[bidx].append(compute_mad(attn_dict[bidx], dist_matrix).cpu().numpy())

    return {b: np.mean(mad_accum[b], axis=0).tolist() for b in all_blocks}


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    mixup_fn,
    bimodal_loss_fn,
    regularized_blocks,
    warmup_factor,
    scaler,
    mode,
):
    model.train()
    total_task_loss = 0.0
    total_reg_loss = 0.0
    total_samples = 0
    reg_info_accum = []

    for images, targets in tqdm(train_loader, desc="Training", leave=False):
        images, targets = images.to(device), targets.to(device)
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            task_loss = criterion(outputs, targets)

        reg_loss = torch.tensor(0.0, device=device)
        reg_info = {}
        if mode == "regularized" and bimodal_loss_fn is not None:
            attn_dict = get_cached_attn_weights(model, regularized_blocks)
            if attn_dict:
                # Bimodal loss computed outside autocast for numerical stability
                reg_loss, reg_info = bimodal_loss_fn(attn_dict, warmup_factor=warmup_factor)
                reg_info_accum.append(reg_info)

        loss = task_loss + reg_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        clear_cached_attn_weights(model, regularized_blocks)

        total_task_loss += task_loss.item() * images.size(0)
        total_reg_loss += reg_loss.item() * images.size(0)
        total_samples += images.size(0)

    avg_task_loss = total_task_loss / max(1, total_samples)
    avg_reg_loss = total_reg_loss / max(1, total_samples)
    return avg_task_loss, avg_reg_loss, reg_info_accum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["baseline", "regularized"])
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.LR)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--debug", action="store_true", help="Use small subset for debugging")
    parser.add_argument("--num_workers", type=int, default=cfg.NUM_WORKERS)
    args = parser.parse_args()

    device = cfg.DEVICE
    mode = args.mode

    if args.output_dir:
        output_dir = args.output_dir
    elif mode == "baseline":
        output_dir = cfg.BASELINE_FT_DIR
    else:
        output_dir = cfg.REGULARIZED_FT_DIR

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    print(f"=" * 60)
    print(f"Bimodal Head Specialization Experiment")
    print(f"Mode: {mode}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"=" * 60)

    # Save args
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Model
    print("Loading pretrained DeiT-S...")
    model = load_deit_small(pretrained=True, device=device)

    regularized_blocks = cfg.REGULARIZED_BLOCKS
    if mode == "regularized":
        # Patch attention forward on regularized blocks to cache attn weights with grads
        patch_attention_forward(model, regularized_blocks, differentiable=True)
        print(f"Patched blocks {regularized_blocks} for attention weight extraction")
    else:
        # For baseline, also patch so we can log MADs but without grad (won't affect training)
        # Actually, don't patch during training for baseline — just patch during eval MAD logging
        pass

    # Data
    data_root = args.data_root or cfg.DATA_ROOT
    if args.debug:
        print("DEBUG MODE: using 10-class subset")
        train_loader, val_loader = get_debug_loaders(data_root, num_classes=10, batch_size=args.batch_size)
    else:
        train_loader = get_train_loader(data_root, batch_size=args.batch_size, num_workers=args.num_workers)
        val_loader = get_val_loader(data_root, batch_size=args.batch_size, num_workers=args.num_workers)

    # Mixup / CutMix
    mixup_fn = Mixup(
        mixup_alpha=cfg.MIXUP_ALPHA,
        cutmix_alpha=cfg.CUTMIX_ALPHA,
        prob=cfg.MIXUP_PROB,
        switch_prob=cfg.MIXUP_SWITCH_PROB,
        num_classes=cfg.NUM_CLASSES,
    )

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    bimodal_loss_fn = BimodalHeadLoss() if mode == "regularized" else None

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs=5, total_epochs=args.epochs)

    # AMP
    scaler = GradScaler()

    # Distance matrix for MAD logging
    dist_matrix = build_distance_matrix(device=device)

    # Training loop
    best_acc1 = 0.0
    epoch_logs = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        warmup_factor = get_warmup_factor(epoch, cfg.WARMUP_EPOCHS)

        avg_task_loss, avg_reg_loss, reg_info = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            mixup_fn=mixup_fn,
            bimodal_loss_fn=bimodal_loss_fn,
            regularized_blocks=regularized_blocks,
            warmup_factor=warmup_factor,
            scaler=scaler,
            mode=mode,
        )

        scheduler.step()
        val_acc1, val_acc5 = validate(model, val_loader, device)
        epoch_mads = compute_epoch_mads(model, val_loader, device, dist_matrix, num_batches=5)

        elapsed = time.time() - t0

        log_entry = {
            "epoch": epoch,
            "train_loss": avg_task_loss,
            "reg_loss": avg_reg_loss,
            "val_acc1": val_acc1,
            "val_acc5": val_acc5,
            "lr": optimizer.param_groups[0]["lr"],
            "warmup_factor": warmup_factor,
            "mads": epoch_mads,
            "time_s": elapsed,
        }

        # Log last reg info if available
        if reg_info:
            last_info = reg_info[-1]
            log_entry["last_reg_info"] = {
                k: v for k, v in last_info.items()
                if isinstance(v, (int, float, str))
            }

        epoch_logs.append(log_entry)

        # Save log incrementally
        with open(os.path.join(output_dir, "training_log.json"), "w") as f:
            json.dump(epoch_logs, f, indent=2)

        # Checkpoint
        is_best = val_acc1 > best_acc1
        if is_best:
            best_acc1 = val_acc1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc1": val_acc1,
                "val_acc5": val_acc5,
            }, os.path.join(output_dir, "checkpoints", "best.pth"))

        # Print summary
        reg_str = f"  reg_loss={avg_reg_loss:.6f}" if mode == "regularized" else ""
        best_str = " *BEST*" if is_best else ""
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"task_loss={avg_task_loss:.4f}{reg_str} | "
              f"val_acc1={val_acc1:.2f}% val_acc5={val_acc5:.2f}% | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | "
              f"{elapsed:.0f}s{best_str}")

    # Save final checkpoint
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc1": val_acc1,
        "val_acc5": val_acc5,
    }, os.path.join(output_dir, "checkpoints", "final.pth"))

    print(f"\nTraining complete. Best val_acc1: {best_acc1:.2f}%")
    print(f"Checkpoints and logs saved to {output_dir}")


if __name__ == "__main__":
    main()
