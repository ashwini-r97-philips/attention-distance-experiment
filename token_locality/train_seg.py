"""ADE20K segmentation training with token locality gates.

Three modes:
  --mode baseline             : finetune DeiT-S decoder, no gate
  --mode token_locality_v3    : + scalar locality gate (per-token per-head signed gate)
  --mode token_locality_v3_vec: + vector locality gate (MLP + RBF basis)

Outputs written to runs/seg/<mode>_seg/ relative to the project root.

Usage (from token_locality/):
  python train_seg.py --mode baseline --epochs 40 --data_root /path/to/ADEChallengeData2016
  python train_seg.py --mode token_locality_v3 --epochs 40 \
      --data_root /path/to/ADEChallengeData2016 \
      --gate_blocks 0 1 2 3 4 5 --gate_lr_multiplier 10.0
  python train_seg.py --mode token_locality_v3_vec --epochs 40 \
      --data_root /path/to/ADEChallengeData2016 \
      --gate_blocks 0 1 2 3 4 5 --gate_lr_multiplier 10.0
"""

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "bimodal_head_specialisation"))

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from seg_model import (
    DeiTSegModel,
    IGNORE_INDEX, NUM_BLOCKS, GRID_H, GRID_W, NUM_PATCHES,
    EMBED_DIM, NUM_HEADS, IMG_SIZE,
)
from seg_data import get_train_loader, get_val_loader
from seg_metrics import compute_miou, compute_boundary_f1
from seg_attention_hooks import capture_attention
from token_locality_gate_v3 import (
    TokenLocalityGateModuleV3,
    VectorKernelGateModule,
    build_patch_distance_matrix,
)

LOCAL_RADIUS_TAU = 5   # patch units (32×32 grid)
WARMUP_EPOCHS    = 10


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_poly_schedule(optimizer, total_epochs, power=0.9, min_lr=1e-7):
    def lr_lambda(epoch):
        factor = (1 - epoch / total_epochs) ** power
        return max(factor, min_lr / optimizer.defaults["lr"])
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_warmup_factor(epoch):
    if epoch >= WARMUP_EPOCHS:
        return 1.0
    return epoch / max(1, WARMUP_EPOCHS)


# ─── Validation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    all_pred, all_target, boundary_f1s = [], [], []

    for images, masks in tqdm(val_loader, desc="Validating", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        pred = model(images, return_aux=False).argmax(dim=1)
        for b in range(pred.shape[0]):
            p, t = pred[b], masks[b]
            all_pred.append(p.cpu())
            all_target.append(t.cpu())
            f1, _, _ = compute_boundary_f1(p, t, ignore_index=IGNORE_INDEX)
            boundary_f1s.append(f1)

    pred_cat = torch.cat([p.flatten() for p in all_pred])
    target_cat = torch.cat([t.flatten() for t in all_target])
    miou, _ = compute_miou(pred_cat, target_cat, num_classes=150,
                           ignore_index=IGNORE_INDEX)
    return miou, float(np.mean(boundary_f1s))


# ─── MAD logging ─────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_epoch_mads(model, val_loader, device, dist_matrix, num_batches=3):
    from seg_metrics import compute_miou  # local import to avoid circular
    from token_locality_gate_v3 import build_patch_distance_matrix

    # import MAD helpers inline to stay self-contained
    def _mad(attn, R):
        a = attn[:, :, 1:, 1:]
        B, H, N, _ = a.shape
        R_ = R[:N, :N].to(a.device)
        return (a * R_.unsqueeze(0).unsqueeze(0)).sum(-1).mean((0, 2))

    def _lm(attn, R, tau):
        a = attn[:, :, 1:, 1:]
        B, H, N, _ = a.shape
        R_ = R[:N, :N].to(a.device)
        return (a * (R_ <= tau).float().unsqueeze(0).unsqueeze(0)).sum(-1).mean((0, 2))

    def _ent(attn):
        a = attn[:, :, 1:, 1:].clamp(min=1e-8)
        return -(a * a.log()).sum(-1).mean((0, 2))

    model.eval()
    all_blocks = list(range(NUM_BLOCKS))
    accum = {b: {"mad": [], "lm": [], "ent": []} for b in all_blocks}

    for batch_idx, (images, _) in enumerate(val_loader):
        if batch_idx >= num_batches:
            break
        images = images.to(device)
        with capture_attention(model.encoder, all_blocks) as get_attn:
            _ = model(images, return_aux=False)
            attn_dict = get_attn()
        for b in all_blocks:
            accum[b]["mad"].append(_mad(attn_dict[b], dist_matrix).cpu().numpy())
            accum[b]["lm"].append(_lm(attn_dict[b], dist_matrix, LOCAL_RADIUS_TAU).cpu().numpy())
            accum[b]["ent"].append(_ent(attn_dict[b]).cpu().numpy())

    return {
        b: {
            "mad": np.mean(accum[b]["mad"], axis=0).tolist(),
            "local_mass": np.mean(accum[b]["lm"], axis=0).tolist(),
            "entropy": np.mean(accum[b]["ent"], axis=0).tolist(),
        }
        for b in all_blocks
    }


# ─── Training loop ───────────────────────────────────────────────────────────

def train_one_epoch(model, train_loader, optimizer, criterion, device, scaler,
                    gate_module, aux_weight=0.4):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, masks in tqdm(train_loader, desc="Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()

        with autocast():
            logits, aux_logits = model(images, return_aux=True)
            loss = (criterion(logits, masks)
                    + aux_weight * criterion(aux_logits, masks))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        all_params = list(model.parameters())
        if gate_module is not None:
            all_params += list(gate_module.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return total_loss / max(1, total_samples)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["baseline", "token_locality_v3", "token_locality_v3_vec"])
    parser.add_argument("--data_root", required=True,
                        help="Path to ADEChallengeData2016/")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--decoder_lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    # Gate args
    parser.add_argument("--gate_blocks", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--gate_lr_multiplier", type=float, default=10.0)
    parser.add_argument("--gate_scale", type=float, default=2.0)
    parser.add_argument("--gate_distance_scale", type=float, default=2.0)
    parser.add_argument("--gate_weight_std", type=float, default=0.02)
    parser.add_argument("--gate_num_basis", type=int, default=16)
    parser.add_argument("--gate_hidden_dim", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = args.mode

    output_dir = args.output_dir or os.path.join(
        _PROJECT_ROOT, "runs", "seg", f"{mode}_seg"
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    print("=" * 60)
    print("Token Locality — ADE20K Segmentation")
    print(f"Mode   : {mode}")
    print(f"Device : {device}")
    print(f"Data   : {args.data_root}")
    print(f"Output : {output_dir}")
    if mode != "baseline":
        print(f"Gate blocks: {args.gate_blocks}  lr_mult: {args.gate_lr_multiplier}")
    print("=" * 60)

    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Model ──
    print("Loading DeiT-S segmentation model...")
    model = DeiTSegModel(pretrained=True).to(device)
    print(f"  Encoder: {sum(p.numel() for p in model.encoder.parameters()) / 1e6:.1f}M params")
    print(f"  Input: {IMG_SIZE}×{IMG_SIZE} → {GRID_H}×{GRID_W} = {NUM_PATCHES} tokens")

    # ── Gate module ──
    gate_module = None

    if mode == "token_locality_v3":
        gate_module = TokenLocalityGateModuleV3(
            model=model.encoder,
            block_indices=args.gate_blocks,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            grid=GRID_H,
            gate_distance_scale=args.gate_distance_scale,
            gate_scale=args.gate_scale,
            device=device,
            weight_std=args.gate_weight_std,
        ).to(device)
        print(f"  TokenLocalityGateV3 on blocks {args.gate_blocks} "
              f"({sum(p.numel() for p in gate_module.parameters())} params)")

    elif mode == "token_locality_v3_vec":
        gate_module = VectorKernelGateModule(
            model=model.encoder,
            block_indices=args.gate_blocks,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            grid=GRID_H,
            gate_distance_scale=args.gate_distance_scale,
            gate_scale=args.gate_scale,
            device=device,
            num_basis=args.gate_num_basis,
            hidden_dim=args.gate_hidden_dim,
            weight_std=args.gate_weight_std,
        ).to(device)
        print(f"  VectorKernelGate on blocks {args.gate_blocks} "
              f"({sum(p.numel() for p in gate_module.parameters())} params)")

    # ── Data ──
    print(f"Loading ADE20K from {args.data_root}...")
    train_loader = get_train_loader(args.data_root, args.batch_size, args.num_workers)
    val_loader = get_val_loader(args.data_root,
                                max(1, args.batch_size // 2), args.num_workers)
    print(f"  Train: {len(train_loader.dataset)} images")
    print(f"  Val:   {len(val_loader.dataset)} images")

    # ── Optimizer ──
    gate_lr = args.backbone_lr * args.gate_lr_multiplier if gate_module is not None else None
    param_groups = model.get_encoder_param_groups(
        args.backbone_lr, args.decoder_lr,
        gate_module=gate_module, gate_lr=gate_lr,
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    scheduler = get_poly_schedule(optimizer, total_epochs=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    scaler = GradScaler()

    # ── Resume ──
    start_epoch = 1
    best_miou = 0.0
    epoch_logs = []

    resume_path = args.resume
    if resume_path is None:
        auto = os.path.join(output_dir, "checkpoints", "best.pth")
        if os.path.exists(auto):
            resume_path = auto

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if gate_module is not None and "gate_state_dict" in ckpt:
            gate_module.load_state_dict(ckpt["gate_state_dict"])
            print("  Gate weights restored.")
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_miou = ckpt.get("val_miou", 0.0)
        for _ in range(start_epoch - 1):
            scheduler.step()
        log_path = os.path.join(output_dir, "training_log.json")
        if os.path.exists(log_path):
            with open(log_path) as f:
                epoch_logs = json.load(f)
        print(f"  Resumed at epoch {start_epoch}, best mIoU: {best_miou:.4f}")

    dist_matrix = build_patch_distance_matrix(GRID_H, device)

    # ── Training loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        avg_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            gate_module=gate_module,
        )

        scheduler.step()
        miou, bf1 = validate(model, val_loader, device)

        epoch_mads = {}
        if epoch % 5 == 0 or epoch == args.epochs or epoch == start_epoch:
            epoch_mads = compute_epoch_mads(model, val_loader, device, dist_matrix)

        elapsed = time.time() - t0

        log_entry = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_miou": miou,
            "val_boundary_f1": bf1,
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_decoder": optimizer.param_groups[1]["lr"],
            "mads": {str(k): v for k, v in epoch_mads.items()},
            "time_s": elapsed,
        }
        if gate_module is not None:
            gate_stats = gate_module.gate_statistics()
            log_entry["gate_stats"] = gate_stats
            print("  [Gate] ", end="")
            for bk, s in gate_stats.items():
                print(f"{bk}: out={s['gate_output_mean']:.4f}±{s['gate_output_std']:.4f}  ",
                      end="")
            print()

        epoch_logs.append(log_entry)
        with open(os.path.join(output_dir, "training_log.json"), "w") as f:
            json.dump(epoch_logs, f, indent=2)

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

        best_str = " *BEST*" if is_best else ""
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"loss={avg_loss:.4f} | "
              f"mIoU={miou:.4f} bF1={bf1:.4f} | "
              f"{elapsed:.0f}s{best_str}")

    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "val_miou": miou,
            "val_boundary_f1": bf1,
            **({"gate_state_dict": gate_module.state_dict()} if gate_module is not None else {}),
        },
        os.path.join(output_dir, "checkpoints", "final.pth"),
    )

    print(f"\nTraining complete. Best mIoU: {best_miou:.4f}")
    print(f"Checkpoints and logs in {output_dir}")


if __name__ == "__main__":
    main()
