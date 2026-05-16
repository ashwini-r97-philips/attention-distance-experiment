"""Segmentation training: DeiT-S + linear decoder on ADE20K.

Two modes:
  --mode baseline     : finetune without regularizer
  --mode regularized  : finetune with bimodal head distance regularizer (10× stronger)

Usage:
  CUDA_VISIBLE_DEVICES=2 python seg_train.py --mode baseline --epochs 40
  CUDA_VISIBLE_DEVICES=2 python seg_train.py --mode regularized --epochs 40
"""

import argparse
import json
import math
import os
import sys
import time

_SEG_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SEG_DIR)
_TOKEN_LOCALITY_DIR = os.path.join(os.path.dirname(_PROJECT_ROOT), "token_locality")

sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _TOKEN_LOCALITY_DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import config as cfg
from model import DeiTSegModel
from common.attention_hooks import (
    patch_attention_forward,
    get_cached_attn_weights,
    clear_cached_attn_weights,
)
from common.bimodal_loss import BimodalHeadLoss
from common.mad_metrics import build_distance_matrix, compute_mad, compute_local_mass, compute_attention_entropy
from data import get_train_loader, get_val_loader
from common.boundary_utils import compute_miou, compute_boundary_f1


def get_poly_schedule(optimizer, total_epochs, power=0.9, min_lr=1e-7):
    """Polynomial LR decay schedule (standard for segmentation)."""
    def lr_lambda(epoch):
        factor = (1 - epoch / total_epochs) ** power
        return max(factor, min_lr / optimizer.defaults["lr"])
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_warmup_factor(epoch, warmup_epochs):
    if epoch >= warmup_epochs:
        return 1.0
    return epoch / max(1, warmup_epochs)


@torch.no_grad()
def validate(model, val_loader, device, num_classes=None):
    """Compute mIoU and boundary F1 on validation set."""
    num_classes = num_classes or cfg.NUM_SEG_CLASSES
    model.eval()

    all_pred = []
    all_target = []
    boundary_f1s = []

    for images, masks in tqdm(val_loader, desc="Validating", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images, return_aux=False)
        pred = logits.argmax(dim=1)  # (B, H, W)

        for b in range(pred.shape[0]):
            p = pred[b]
            t = masks[b]
            all_pred.append(p.cpu())
            all_target.append(t.cpu())
            f1, _, _ = compute_boundary_f1(p, t, ignore_index=cfg.IGNORE_INDEX)
            boundary_f1s.append(f1)

    # Compute overall mIoU
    pred_cat = torch.cat([p.flatten() for p in all_pred])
    target_cat = torch.cat([t.flatten() for t in all_target])
    miou, per_class = compute_miou(pred_cat, target_cat, num_classes, ignore_index=cfg.IGNORE_INDEX)
    mean_bf1 = float(np.mean(boundary_f1s))

    return miou, mean_bf1, per_class


@torch.no_grad()
def compute_epoch_mads(model, val_loader, device, dist_matrix, num_batches=5):
    """Log per-block MAD, local mass, entropy on a few val batches."""
    from common.attention_hooks import capture_attention
    model.eval()
    all_blocks = list(range(cfg.NUM_BLOCKS))
    accum = {b: {"mad": [], "lm": [], "ent": []} for b in all_blocks}

    for batch_idx, (images, _) in enumerate(val_loader):
        if batch_idx >= num_batches:
            break
        images = images.to(device)
        # Need to capture attention from encoder blocks
        with capture_attention(model.encoder, all_blocks) as get_attn:
            _ = model(images, return_aux=False)
            attn_dict = get_attn()
        for bidx in all_blocks:
            accum[bidx]["mad"].append(compute_mad(attn_dict[bidx], dist_matrix).cpu().numpy())
            accum[bidx]["lm"].append(compute_local_mass(
                attn_dict[bidx], dist_matrix,
                tau=cfg.LOCAL_RADIUS_TAU,
            ).cpu().numpy())
            accum[bidx]["ent"].append(compute_attention_entropy(attn_dict[bidx]).cpu().numpy())

    result = {}
    for b in all_blocks:
        result[b] = {
            "mad": np.mean(accum[b]["mad"], axis=0).tolist(),
            "local_mass": np.mean(accum[b]["lm"], axis=0).tolist(),
            "entropy": np.mean(accum[b]["ent"], axis=0).tolist(),
        }
    return result


def train_one_epoch(
    model, train_loader, optimizer, criterion, device,
    bimodal_loss_fn, regularized_blocks, warmup_factor,
    scaler, mode, aux_weight=0.4,
):
    model.train()
    total_task_loss = 0.0
    total_reg_loss = 0.0
    total_samples = 0
    reg_info_accum = []

    for images, masks in tqdm(train_loader, desc="Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        with autocast():
            logits, aux_logits = model(images, return_aux=True)
            main_loss = criterion(logits, masks)
            aux_loss = criterion(aux_logits, masks)
            task_loss = main_loss + aux_weight * aux_loss

        reg_loss = torch.tensor(0.0, device=device)
        reg_info = {}
        if mode == "regularized" and bimodal_loss_fn is not None:
            attn_dict = get_cached_attn_weights(model.encoder, regularized_blocks)
            if attn_dict:
                reg_loss, reg_info = bimodal_loss_fn(attn_dict, warmup_factor=warmup_factor)
                reg_info_accum.append(reg_info)

        loss = task_loss + reg_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        clear_cached_attn_weights(model.encoder, regularized_blocks)

        total_task_loss += task_loss.item() * images.size(0)
        total_reg_loss += reg_loss.item() * images.size(0)
        total_samples += images.size(0)

    avg_task_loss = total_task_loss / max(1, total_samples)
    avg_reg_loss = total_reg_loss / max(1, total_samples)
    return avg_task_loss, avg_reg_loss, reg_info_accum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        choices=["baseline", "regularized", "gaussian_bias", "token_locality_v3"])
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--backbone_lr", type=float, default=cfg.BACKBONE_LR)
    parser.add_argument("--decoder_lr", type=float, default=cfg.DECODER_LR)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=cfg.NUM_WORKERS)
    parser.add_argument("--lambda_gap", type=float, default=cfg.LAMBDA_GAP)
    parser.add_argument("--lambda_compact", type=float, default=cfg.LAMBDA_COMPACT)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g. checkpoints/best.pth)")
    # Gate options (gaussian_bias and token_locality_v3 modes)
    parser.add_argument("--gate_blocks", type=int, nargs="+", default=list(range(0, 6)),
                        help="Transformer block indices to install gates on")
    parser.add_argument("--gate_lr_multiplier", type=float, default=10.0,
                        help="Gate LR = backbone_lr * gate_lr_multiplier")
    parser.add_argument("--gate_scale", type=float, default=2.0,
                        help="tanh ceiling for v3 gate values")
    parser.add_argument("--gate_distance_scale", type=float, default=2.0,
                        help="Multiplier on distance penalty before softmax (v3)")
    parser.add_argument("--gate_weight_std", type=float, default=0.02,
                        help="Init std for gate linear weights")
    parser.add_argument("--gaussian_init_sigma", type=float, default=0.25,
                        help="Initial σ for LearnedGaussianBias")
    args = parser.parse_args()

    device = cfg.DEVICE
    mode = args.mode

    if args.output_dir:
        output_dir = args.output_dir
    elif mode == "baseline":
        output_dir = cfg.BASELINE_SEG_DIR
    elif mode == "regularized":
        output_dir = cfg.REGULARIZED_SEG_DIR
    elif mode == "gaussian_bias":
        output_dir = os.path.join(cfg.OUTPUT_DIR, "gaussian_bias_seg")
    else:  # token_locality_v3
        output_dir = os.path.join(cfg.OUTPUT_DIR, "token_v3_seg")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    print("=" * 60)
    print("Bimodal Head Specialization — ADE20K Segmentation")
    print(f"Mode: {mode}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    if mode == "regularized":
        print(f"λ_gap={args.lambda_gap}, λ_compact={args.lambda_compact}, warmup={cfg.WARMUP_EPOCHS} epochs")
    if mode in ("gaussian_bias", "token_locality_v3"):
        print(f"Gate blocks: {args.gate_blocks}, gate_lr_multiplier={args.gate_lr_multiplier}")
    print("=" * 60)

    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Model
    print("Loading DeiT-S segmentation model...")
    model = DeiTSegModel(pretrained=True).to(device)
    print(f"  Encoder: {sum(p.numel() for p in model.encoder.parameters()) / 1e6:.1f}M params")
    print(f"  Decoder: {sum(p.numel() for p in model.decoder.parameters()) / 1e6:.1f}M params")
    print(f"  Input: {cfg.IMG_SIZE}×{cfg.IMG_SIZE} → {cfg.GRID_H}×{cfg.GRID_W} = {cfg.NUM_PATCHES} tokens")

    regularized_blocks = cfg.REGULARIZED_BLOCKS
    gate_module = None

    if mode == "regularized":
        patch_attention_forward(model.encoder, regularized_blocks, differentiable=True)
        print(f"  Patched blocks {regularized_blocks} for differentiable attention capture")

    elif mode == "gaussian_bias":
        from gate_modules_seg import LearnedGaussianBiasModule
        gate_module = LearnedGaussianBiasModule(
            model=model,
            block_indices=args.gate_blocks,
            num_heads=cfg.NUM_HEADS,
            grid=cfg.GRID_H,
            device=device,
            init_sigma=args.gaussian_init_sigma,
        ).to(device)
        n_gate = sum(p.numel() for p in gate_module.parameters())
        print(f"  LearnedGaussianBias installed on blocks {args.gate_blocks} ({n_gate} gate params)")

    elif mode == "token_locality_v3":
        from token_locality_gate_v3 import TokenLocalityGateModuleV3
        gate_module = TokenLocalityGateModuleV3(
            model=model,
            block_indices=args.gate_blocks,
            embed_dim=cfg.EMBED_DIM,
            num_heads=cfg.NUM_HEADS,
            grid=cfg.GRID_H,
            gate_distance_scale=args.gate_distance_scale,
            gate_scale=args.gate_scale,
            device=device,
            weight_std=args.gate_weight_std,
        ).to(device)
        n_gate = sum(p.numel() for p in gate_module.parameters())
        print(f"  TokenLocalityGateV3 installed on blocks {args.gate_blocks} ({n_gate} gate params)")

    # Data
    print(f"Loading ADE20K from {cfg.DATA_ROOT}...")
    train_loader = get_train_loader(batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = get_val_loader(batch_size=max(1, args.batch_size // 2), num_workers=args.num_workers)
    print(f"  Train: {len(train_loader.dataset)} images, {len(train_loader)} batches")
    print(f"  Val: {len(val_loader.dataset)} images")

    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.IGNORE_INDEX)
    bimodal_loss_fn = None
    if mode == "regularized":
        bimodal_loss_fn = BimodalHeadLoss(
            lambda_gap=args.lambda_gap,
            lambda_compact=args.lambda_compact,
        )
        # Override the distance matrix to use 32×32 grid
        bimodal_loss_fn._dist_matrix = build_distance_matrix(
            grid_h=cfg.GRID_H, grid_w=cfg.GRID_W, device=device
        )

    # Optimizer with differential LR
    gate_lr = args.backbone_lr * args.gate_lr_multiplier if gate_module is not None else None
    param_groups = model.get_encoder_param_groups(args.backbone_lr, args.decoder_lr,
                                                  gate_module=gate_module, gate_lr=gate_lr)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = get_poly_schedule(optimizer, total_epochs=args.epochs)

    scaler = GradScaler()

    # Resume from checkpoint
    start_epoch = 1
    best_miou = 0.0
    epoch_logs = []

    resume_path = args.resume
    if resume_path is None:
        # Auto-detect: if best.pth exists in output_dir, resume from it
        auto_path = os.path.join(output_dir, "checkpoints", "best.pth")
        if os.path.exists(auto_path):
            resume_path = auto_path

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_miou = ckpt.get("val_miou", 0.0)
        # Advance scheduler to correct position
        for _ in range(start_epoch - 1):
            scheduler.step()
        # Reload existing logs
        log_path = os.path.join(output_dir, "training_log.json")
        if os.path.exists(log_path):
            with open(log_path) as f:
                epoch_logs = json.load(f)
        print(f"  Resumed at epoch {start_epoch}, best mIoU so far: {best_miou:.4f}")

    # Distance matrix for MAD logging (32×32 grid)
    dist_matrix = build_distance_matrix(grid_h=cfg.GRID_H, grid_w=cfg.GRID_W, device=device)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        warmup_factor = get_warmup_factor(epoch, cfg.WARMUP_EPOCHS)

        avg_task_loss, avg_reg_loss, reg_info = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            bimodal_loss_fn=bimodal_loss_fn,
            regularized_blocks=regularized_blocks,
            warmup_factor=warmup_factor,
            scaler=scaler,
            mode=mode,
        )

        scheduler.step()

        # Validate
        miou, bf1, per_class = validate(model, val_loader, device)

        # Log MADs (every 5 epochs or last epoch — expensive with 1024 tokens)
        epoch_mads = {}
        if epoch % 5 == 0 or epoch == args.epochs or epoch == start_epoch:
            epoch_mads = compute_epoch_mads(model, val_loader, device, dist_matrix, num_batches=3)

        elapsed = time.time() - t0

        log_entry = {
            "epoch": epoch,
            "train_loss": avg_task_loss,
            "reg_loss": avg_reg_loss,
            "val_miou": miou,
            "val_boundary_f1": bf1,
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_decoder": optimizer.param_groups[1]["lr"],
            "warmup_factor": warmup_factor,
            "mads": {str(k): v for k, v in epoch_mads.items()},
            "time_s": elapsed,
        }
        if reg_info:
            last = reg_info[-1]
            log_entry["last_reg_info"] = {k: v for k, v in last.items() if isinstance(v, (int, float, str))}
        if gate_module is not None:
            log_entry["gate_stats"] = gate_module.gate_statistics()

        epoch_logs.append(log_entry)
        with open(os.path.join(output_dir, "training_log.json"), "w") as f:
            json.dump(epoch_logs, f, indent=2)

        # Checkpoint
        is_best = miou > best_miou
        if is_best:
            best_miou = miou
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_miou": miou,
                "val_boundary_f1": bf1,
            }
            if gate_module is not None:
                ckpt["gate_state_dict"] = gate_module.state_dict()
            torch.save(ckpt, os.path.join(output_dir, "checkpoints", "best.pth"))

        reg_str = f"  reg_loss={avg_reg_loss:.6f}" if mode == "regularized" else ""
        best_str = " *BEST*" if is_best else ""
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"loss={avg_task_loss:.4f}{reg_str} | "
              f"mIoU={miou:.4f} bF1={bf1:.4f} | "
              f"{elapsed:.0f}s{best_str}")

    # Final checkpoint
    final_ckpt = {
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "val_miou": miou,
        "val_boundary_f1": bf1,
    }
    if gate_module is not None:
        final_ckpt["gate_state_dict"] = gate_module.state_dict()
    torch.save(final_ckpt, os.path.join(output_dir, "checkpoints", "final.pth"))

    print(f"\nTraining complete. Best mIoU: {best_miou:.4f}")
    print(f"Checkpoints and logs saved to {output_dir}")


if __name__ == "__main__":
    main()
