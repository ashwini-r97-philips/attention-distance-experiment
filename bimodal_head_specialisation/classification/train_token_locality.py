"""Training script for the token-conditioned locality gate experiment.

DO NOT run this automatically. Start explicitly with:

  python train_token_locality.py \
      --config configs/token_locality_minimal_vit_s16.yaml

This script:
  1. Loads a pretrained ViT-S/16 (same starting point as baseline runs).
  2. Installs per-token locality gate branches into blocks 0-7.
  3. Fine-tunes for 5 epochs with AdamW, AMP, and gradient accumulation.
  4. Logs gate statistics, MAD, local mass, and entropy every epoch.
  5. After training, optionally runs fast localization eval on the 2000-image
     subset and prints a decision table vs bimodal_medium.

Usage
-----
  python train_token_locality.py --config configs/token_locality_minimal_vit_s16.yaml
  python train_token_locality.py --config configs/token_locality_minimal_vit_s16.yaml \
      --resume runs/token_locality_minimal_vit_s16_imagenet1k/checkpoints/epoch_0003.pth
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from common.config import load_config, save_config
from common.model_utils import load_vit_small
from common.attention_hooks import (
    get_cached_attn_weights,
    _attention_forward_with_weights,
)
from common.mad_metrics import (
    build_distance_matrix,
    compute_mad,
    compute_local_mass,
    compute_attention_entropy,
)
from common.token_locality_gate import TokenLocalityGateModule
from data import get_train_loader, get_val_loader, get_attention_eval_subset


# ─── Schedules ───────────────────────────────────────────────────────────────

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


# ─── Attention stats ─────────────────────────────────────────────────────────

@torch.no_grad()
def compute_attention_stats(model, loader, device, dist_matrix, cfg, gate_module):
    """Compute attention stats using gate-installed caching forwards.

    NOTE: The TokenLocalityGateModule has already patched model.blocks[i].attn.forward
    for gate blocks to cache _cached_attn_weights. We must NOT use capture_attention /
    unpatch_attention_forward because that restores the timm class Attention.forward and
    destroys the gate installation.

    For non-gate blocks we install the weight-caching forward temporarily, being careful
    to restore ONLY those blocks without touching gate blocks.
    """
    model.eval()
    all_blocks     = list(range(cfg.num_blocks))
    gate_block_set = set(gate_module.block_indices)
    non_gate_blks  = [b for b in all_blocks if b not in gate_block_set]

    # Temporarily patch non-gate blocks with weight-caching forward
    non_gate_originals: dict = {}
    for b in non_gate_blks:
        attn_module = model.blocks[b].attn
        non_gate_originals[b] = attn_module.forward
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
            images = images.to(device)
            _ = model(images)  # all blocks now cache _cached_attn_weights
            attn_dict = get_cached_attn_weights(model, all_blocks)  # {b: (B,H,N,N)}
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
        # Restore non-gate blocks to their original (unfused SDPA) forwards
        for b in non_gate_blks:
            attn_module = model.blocks[b].attn
            attn_module.forward = non_gate_originals[b]
            attn_module.fused_attn = True

    stats = {}
    for b in all_blocks:
        stats[b] = {}
        for key in accum[b]:
            if accum[b][key]:
                stats[b][key] = np.mean(accum[b][key], axis=0).tolist()
    return stats


# ─── Training ────────────────────────────────────────────────────────────────

def train_one_epoch(
    model,
    gate_module: TokenLocalityGateModule,
    train_loader,
    optimizer,
    criterion,
    device,
    scaler,
    cfg,
    grad_accum_steps: int = 1,
):
    model.train()
    total_task = total_samples = 0.0
    total_grad_norm = 0.0
    n_steps = 0
    optimizer.zero_grad()

    for step, (images, targets) in enumerate(train_loader):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            outputs   = model(images)
            task_loss = criterion(outputs, targets)
            loss      = task_loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(gate_module.parameters()),
                max_norm=cfg.grad_clip_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_grad_norm += gn.item()
            n_steps += 1

        total_task    += task_loss.item() * images.size(0)
        total_samples += images.size(0)

    return {
        "train_loss": total_task / max(1, total_samples),
        "grad_norm":  total_grad_norm / max(1, n_steps),
    }


# ─── Gate statistics logging ─────────────────────────────────────────────────

def log_gate_statistics(gate_module: TokenLocalityGateModule, epoch: int, output_dir: str):
    stats = gate_module.gate_statistics()
    path  = os.path.join(output_dir, "gate_stats", f"epoch_{epoch:04d}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n  [Gate stats]")
    for block_key, s in stats.items():
        print(f"    {block_key}: mean={s['weight_mean']:+.5f}  std={s['weight_std']:.5f}  "
              f"bias={s['bias']:+.5f}")
    return stats


# ─── Post-training localization eval ─────────────────────────────────────────

def run_post_training_localization(cfg, output_dir: str, device_str: str):
    """Run fast localization eval on the 2000-image subset after training."""
    best_ckpt = os.path.join(output_dir, "checkpoints", "best.pth")
    if not os.path.exists(best_ckpt):
        print("[WARNING] No best.pth found, skipping post-training localization eval.")
        return

    baseline_ckpt = getattr(cfg, "localization_eval_baseline_checkpoint",
                             "runs/baseline_vit_s16_imagenet1k/checkpoints/best.pth")
    if not os.path.exists(baseline_ckpt):
        print(f"[WARNING] Baseline checkpoint not found: {baseline_ckpt}")
        return

    ann_dir   = getattr(cfg, "localization_eval_annotation_dir", "data/imagenet_loc_annotations")
    n_images  = getattr(cfg, "localization_eval_num_images", 2000)
    seed      = getattr(cfg, "localization_eval_seed", 42)
    loc_out   = os.path.join(output_dir, "localization_eval_post_train")

    print(f"\n[INFO] Running post-training localization eval ({n_images} images)...")
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "eval_imagenet_localization_fast.py"),
        "--checkpoints",
        f"baseline:{baseline_ckpt}",
        f"token_locality:{best_ckpt}",
        "--annotation-dir", ann_dir,
        "--max-images", str(n_images),
        "--seed", str(seed),
        "--batch-size", "64",
        "--output-dir", loc_out,
        "--amp",
        "--skip-debug",
        "--num-vis", "30",
        f"--device", device_str,
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print("[WARNING] Post-training localization eval failed.")


# ─── Decision summary ────────────────────────────────────────────────────────

def print_decision_summary(
    val_metrics: dict,
    attn_stats: dict,
    gate_stats: dict,
    loc_summary_path: str,
    cfg,
):
    """Print the comparison summary against bimodal_medium."""
    print("\n" + "=" * 65)
    print("TOKEN LOCALITY EXPERIMENT — DECISION SUMMARY")
    print("=" * 65)
    print(f"  Top1 accuracy: {val_metrics['val_acc1']:.2f}%")
    print(f"  Top5 accuracy: {val_metrics['val_acc5']:.2f}%")

    # MAD shift vs baseline (rough: baseline last-layer MAD ~0.29 for ViT-S)
    BASELINE_MAD_APPROX = 0.29
    last_block_mad = attn_stats.get(11, {}).get("mad", None)
    if last_block_mad is not None:
        mean_mad = float(np.mean(last_block_mad))
        print(f"  Mean MAD (last block, all heads): {mean_mad:.4f}  "
              f"(baseline approx: {BASELINE_MAD_APPROX:.4f})")

    # Gate std per layer
    print("\n  Gate weight std per block:")
    for block_key, s in gate_stats.items():
        print(f"    {block_key}: std={s['weight_std']:.5f}  mean={s['weight_mean']:+.5f}")

    # Localization results
    if os.path.exists(loc_summary_path):
        import csv
        with open(loc_summary_path) as f:
            rows = list(csv.DictReader(f))
        print("\n  Localization (post-training, rollout):")
        print(f"  {'Checkpoint':<20} {'PG%':>7} {'CL@0.3':>8} {'CL@0.5':>8}")
        print("  " + "-" * 45)
        for row in rows:
            if row.get("method", "") != "rollout":
                continue
            print(f"  {row['checkpoint']:<20} "
                  f"{float(row['pointing_game_accuracy'])*100:>7.1f} "
                  f"{float(row.get('corloc_thr0.3', 0))*100:>8.1f} "
                  f"{float(row.get('corloc_thr0.5', 0))*100:>8.1f}")

        # Bimodal medium reference values (from earlier 2k eval)
        print("\n  Reference (bimodal_medium, 2k subset):")
        print(f"  {'bimodal_medium':<20} {'53.8':>7} {'28.0':>8} {'0.7':>8}")

        # Compare
        token_loc_row = next(
            (r for r in rows if r.get("checkpoint") == "token_locality"
             and r.get("method", "") == "rollout"), None
        )
        if token_loc_row:
            pg   = float(token_loc_row["pointing_game_accuracy"]) * 100
            cl3  = float(token_loc_row.get("corloc_thr0.3", 0)) * 100
            cl5  = float(token_loc_row.get("corloc_thr0.5", 0)) * 100
            bigger = pg > 53.8 or cl3 > 28.0 or cl5 > 0.7
            print(f"\n  Token-locality produced LARGER effect than bimodal_medium: "
                  f"{'YES' if bigger else 'NO'}")
    else:
        print("\n  [Localization summary not found — run localization eval manually]")

    print("=" * 65)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Token-locality gate training. DO NOT run automatically."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.device)

    output_dir = cfg.output_dir
    assert output_dir, "output_dir must be set in config"
    for subdir in ["checkpoints", "attention_stats", "gate_stats"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    save_config(cfg, os.path.join(output_dir, "config.yaml"))

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = "unknown"
    with open(os.path.join(output_dir, "git_commit.txt"), "w") as f:
        f.write(commit + "\n")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    print("=" * 65)
    print("TOKEN LOCALITY GATE EXPERIMENT")
    print(f"  epochs:           {cfg.epochs}")
    print(f"  batch_size:       {cfg.batch_size}")
    print(f"  lr:               {cfg.lr}")
    print(f"  gate_blocks:      {cfg.regularized_blocks}")
    print(f"  gate_init_scale:  {getattr(cfg, 'gate_init_scale', 0.01)}")
    print(f"  gate_dist_scale:  {getattr(cfg, 'gate_distance_scale', 4.0)}")
    print(f"  output:           {output_dir}")
    print("=" * 65)

    # ── Model ──
    print("Loading pretrained ViT-S/16...")
    model = load_vit_small(cfg, pretrained=True)

    # ── Install locality gates ──
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
    print(f"  Installed gate branches on blocks: {cfg.regularized_blocks}")
    n_gate_params = sum(p.numel() for p in gate_module.parameters())
    print(f"  Gate parameters: {n_gate_params:,}")

    # ── Data ──
    print("Loading ImageNet (streaming)...")
    train_loader = get_train_loader(cfg)
    val_loader   = get_val_loader(cfg, batch_size=max(cfg.batch_size, 256))
    attn_eval_loader, attn_eval_indices = get_attention_eval_subset(cfg)
    with open(os.path.join(output_dir, "attn_eval_indices.json"), "w") as f:
        json.dump(attn_eval_indices, f)

    # ── Loss, optimizer, scaler ──
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # Separate LR for gate branch (same lr, but isolated param group for easy tuning)
    model_params = list(model.parameters())
    gate_params  = list(gate_module.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": model_params, "lr": cfg.lr},
            {"params": gate_params,  "lr": cfg.lr},
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = cosine_schedule_with_warmup(optimizer, cfg.warmup_epochs, cfg.epochs)
    scaler    = torch.amp.GradScaler("cuda")

    # Gradient accumulation: target effective batch ~512 on L4
    grad_accum = max(1, 512 // cfg.batch_size)
    print(f"  Gradient accumulation steps: {grad_accum}  "
          f"(effective batch: {cfg.batch_size * grad_accum})")

    # Distance matrix for attention stats
    dist_matrix = build_distance_matrix(cfg.grid_h, cfg.grid_w, device=device)

    # ── Resume ──
    start_epoch = 1
    best_acc1   = 0.0
    epoch_logs  = []
    final_gate_stats = {}
    final_attn_stats = {}

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "gate_state_dict" in ckpt:
            gate_module.load_state_dict(ckpt["gate_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc1   = ckpt.get("val_acc1", 0.0)
        for _ in range(start_epoch - 1):
            scheduler.step()
        log_path = os.path.join(output_dir, "training_log.json")
        if os.path.exists(log_path):
            with open(log_path) as f:
                epoch_logs = json.load(f)
        print(f"  Resumed at epoch {start_epoch}, best acc1={best_acc1:.2f}%")

    # ─── Training loop ────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        print(f"\n{'─'*55}")
        print(f"Epoch {epoch}/{cfg.epochs}  lr={scheduler.get_last_lr()[0]:.2e}")

        # Retry on streaming errors
        for attempt in range(1, 4):
            try:
                train_metrics = train_one_epoch(
                    model=model,
                    gate_module=gate_module,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    scaler=scaler,
                    cfg=cfg,
                    grad_accum_steps=grad_accum,
                )
                break
            except RuntimeError as e:
                err = str(e)
                if attempt < 3 and any(kw in err.lower() for kw in
                                       ["client has been closed", "connection reset", "timed out"]):
                    print(f"  ⚠ Streaming error (attempt {attempt}/3), retrying: {err[:100]}")
                    train_loader = get_train_loader(cfg)
                    time.sleep(5 * attempt)
                else:
                    raise

        scheduler.step()
        val_metrics = validate(model, val_loader, criterion, device)

        # Attention stats
        attn_stats = compute_attention_stats(model, attn_eval_loader, device, dist_matrix, cfg, gate_module)
        attn_path  = os.path.join(output_dir, "attention_stats", f"epoch_{epoch:04d}.json")
        with open(attn_path, "w") as f:
            json.dump(attn_stats, f)

        # Gate statistics
        gate_stats = log_gate_statistics(gate_module, epoch, output_dir)

        elapsed = time.time() - t0
        print(f"\n  train_loss={train_metrics['train_loss']:.4f}  "
              f"val_acc1={val_metrics['val_acc1']:.2f}%  "
              f"val_acc5={val_metrics['val_acc5']:.2f}%  "
              f"time={elapsed:.0f}s")

        # Log
        epoch_log = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "lr": scheduler.get_last_lr()[0],
            "elapsed_s": round(elapsed, 1),
        }
        epoch_logs.append(epoch_log)
        with open(os.path.join(output_dir, "training_log.json"), "w") as f:
            json.dump(epoch_logs, f, indent=2)

        # Checkpoint every epoch
        ckpt_data = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "gate_state_dict":      gate_module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc1":             val_metrics["val_acc1"],
        }
        ckpt_path = os.path.join(output_dir, "checkpoints", f"epoch_{epoch:04d}.pth")
        torch.save(ckpt_data, ckpt_path)

        if val_metrics["val_acc1"] > best_acc1:
            best_acc1 = val_metrics["val_acc1"]
            torch.save(ckpt_data, os.path.join(output_dir, "checkpoints", "best.pth"))
            print(f"  ★ New best: {best_acc1:.2f}%")

        final_gate_stats = gate_stats
        final_attn_stats = attn_stats

    # ─── Post-training localization eval ──────────────────────────────────
    run_loc = getattr(cfg, "run_localization_eval_after_training", True)
    if run_loc:
        run_post_training_localization(cfg, output_dir, str(device))

    # ─── Decision summary ─────────────────────────────────────────────────
    final_val = validate(model, val_loader, criterion, device)
    loc_summary_path = os.path.join(
        output_dir, "localization_eval_post_train", "metrics_summary.csv"
    )
    print_decision_summary(
        val_metrics=final_val,
        attn_stats=final_attn_stats,
        gate_stats=final_gate_stats,
        loc_summary_path=loc_summary_path,
        cfg=cfg,
    )

    print(f"\n[DONE] Token locality experiment complete.")
    print(f"       Results → {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
