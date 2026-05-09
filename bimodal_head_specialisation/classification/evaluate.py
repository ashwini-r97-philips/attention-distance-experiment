"""Post-training evaluation: GMM bimodality, role persistence, head masking."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from common import config as cfg
from common.model_utils import load_deit_small, mask_heads
from common.attention_hooks import capture_attention
from common.mad_metrics import (
    build_distance_matrix,
    compute_mad,
    compute_non_self_mad,
    compute_local_mass,
    compute_attention_entropy,
)
from data import get_val_loader


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def compute_all_metrics(model, val_loader, device, dist_matrix, num_batches):
    """Compute MAD, non-self MAD, local mass, entropy on val set."""
    model.eval()
    all_blocks = list(range(cfg.NUM_BLOCKS))
    accum = {b: {"mad": [], "ns_mad": [], "lm": [], "ent": []} for b in all_blocks}

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(val_loader, total=num_batches, desc="Metrics")):
            if batch_idx >= num_batches:
                break
            images = images.to(device)
            with capture_attention(model, all_blocks) as get_attn:
                _ = model(images)
                attn_dict = get_attn()
            for bidx in all_blocks:
                attn = attn_dict[bidx]
                accum[bidx]["mad"].append(compute_mad(attn, dist_matrix).cpu().numpy())
                accum[bidx]["ns_mad"].append(compute_non_self_mad(attn, dist_matrix).cpu().numpy())
                accum[bidx]["lm"].append(compute_local_mass(attn, dist_matrix).cpu().numpy())
                accum[bidx]["ent"].append(compute_attention_entropy(attn).cpu().numpy())

    result = {}
    for b in all_blocks:
        result[b] = {
            "mad": np.mean(accum[b]["mad"], axis=0),
            "non_self_mad": np.mean(accum[b]["ns_mad"], axis=0),
            "local_mass": np.mean(accum[b]["lm"], axis=0),
            "entropy": np.mean(accum[b]["ent"], axis=0),
        }
    return result


def gmm_bimodality_test(model, val_loader, device, dist_matrix, num_batches):
    """Fit 1-component and 2-component GMM to headwise MAD per block.

    Collect per-batch MAD values for each head, then fit GMMs to the
    flattened distribution of head MADs within each block.
    """
    model.eval()
    early_blocks = cfg.REGULARIZED_BLOCKS
    mad_per_batch = {b: [] for b in early_blocks}

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(val_loader, total=num_batches, desc="GMM data")):
            if batch_idx >= num_batches:
                break
            images = images.to(device)
            with capture_attention(model, early_blocks) as get_attn:
                _ = model(images)
                attn_dict = get_attn()
            for bidx in early_blocks:
                mad_per_batch[bidx].append(compute_mad(attn_dict[bidx], dist_matrix).cpu().numpy())

    results = {}
    for bidx in early_blocks:
        # Shape: (num_batches, num_heads) → flatten to (num_batches * num_heads,)
        all_mads = np.stack(mad_per_batch[bidx], axis=0)  # (num_batches, num_heads)
        flat = all_mads.flatten().reshape(-1, 1)

        gmm1 = GaussianMixture(n_components=1, random_state=42).fit(flat)
        gmm2 = GaussianMixture(n_components=2, random_state=42).fit(flat)

        results[bidx] = {
            "bic_1comp": gmm1.bic(flat),
            "bic_2comp": gmm2.bic(flat),
            "bic_diff": gmm1.bic(flat) - gmm2.bic(flat),  # positive = 2-comp better
            "gmm2_means": gmm2.means_.flatten().tolist(),
            "gmm2_weights": gmm2.weights_.tolist(),
            "per_head_mean_mad": all_mads.mean(axis=0).tolist(),
        }

    return results


def role_persistence(model, val_loader, device, dist_matrix, num_batches):
    """Check how stable each head's local/global role is across batches."""
    model.eval()
    early_blocks = cfg.REGULARIZED_BLOCKS
    assignments = {b: [] for b in early_blocks}

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(val_loader, total=num_batches, desc="Persistence")):
            if batch_idx >= num_batches:
                break
            images = images.to(device)
            with capture_attention(model, early_blocks) as get_attn:
                _ = model(images)
                attn_dict = get_attn()
            for bidx in early_blocks:
                mads = compute_mad(attn_dict[bidx], dist_matrix).cpu().numpy()  # (H,)
                median = np.median(mads)
                # 0 = local (below median), 1 = global (above median)
                assignment = (mads > median).astype(int)
                assignments[bidx].append(assignment)

    results = {}
    for bidx in early_blocks:
        all_assignments = np.stack(assignments[bidx], axis=0)  # (num_batches, num_heads)
        # For each head, fraction of batches where it's in its most common role
        persistence_per_head = []
        for h in range(cfg.NUM_HEADS):
            counts = np.bincount(all_assignments[:, h], minlength=2)
            persistence_per_head.append(counts.max() / counts.sum())
        results[bidx] = {
            "persistence_per_head": persistence_per_head,
            "mean_persistence": float(np.mean(persistence_per_head)),
            "dominant_role": [int(np.argmax(np.bincount(all_assignments[:, h], minlength=2)))
                             for h in range(cfg.NUM_HEADS)],
        }

    return results


@torch.no_grad()
def validate_accuracy(model, val_loader, device):
    model.eval()
    correct1 = 0
    total = 0
    for images, targets in val_loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        _, pred = outputs.topk(1, dim=1)
        correct1 += (pred.squeeze(1) == targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct1 / total


def head_masking_experiment(model, val_loader, device, dist_matrix, num_batches=20):
    """Zero out local, global, and random heads; measure accuracy impact."""
    model.eval()

    # First, determine head roles from val MADs
    from common.attention_hooks import capture_attention as ca
    early_blocks = cfg.REGULARIZED_BLOCKS
    mad_accum = {b: [] for b in early_blocks}

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(val_loader):
            if batch_idx >= num_batches:
                break
            images = images.to(device)
            with ca(model, early_blocks) as get_attn:
                _ = model(images)
                attn_dict = get_attn()
            for bidx in early_blocks:
                mad_accum[bidx].append(compute_mad(attn_dict[bidx], dist_matrix).cpu().numpy())

    head_roles = {}
    for bidx in early_blocks:
        avg_mad = np.mean(mad_accum[bidx], axis=0)
        sorted_heads = np.argsort(avg_mad)
        head_roles[bidx] = {
            "local_heads": sorted_heads[:cfg.NUM_LOW_HEADS].tolist(),
            "global_heads": sorted_heads[cfg.NUM_LOW_HEADS:].tolist(),
        }

    # Baseline accuracy (no masking)
    no_mask_acc = validate_accuracy(model, val_loader, device)

    # Mask local heads across all early blocks
    mask_contexts = []
    for bidx in early_blocks:
        ctx = mask_heads(model, bidx, head_roles[bidx]["local_heads"])
        mask_contexts.append(ctx)
    for ctx in mask_contexts:
        ctx.__enter__()
    mask_local_acc = validate_accuracy(model, val_loader, device)
    for ctx in reversed(mask_contexts):
        ctx.__exit__(None, None, None)

    # Mask global heads across all early blocks
    mask_contexts = []
    for bidx in early_blocks:
        ctx = mask_heads(model, bidx, head_roles[bidx]["global_heads"])
        mask_contexts.append(ctx)
    for ctx in mask_contexts:
        ctx.__enter__()
    mask_global_acc = validate_accuracy(model, val_loader, device)
    for ctx in reversed(mask_contexts):
        ctx.__exit__(None, None, None)

    # Mask random heads (average over trials)
    random_accs = []
    rng = np.random.RandomState(42)
    for trial in range(cfg.NUM_MASK_TRIALS):
        mask_contexts = []
        for bidx in early_blocks:
            random_heads = rng.choice(cfg.NUM_HEADS, size=cfg.NUM_LOW_HEADS, replace=False).tolist()
            ctx = mask_heads(model, bidx, random_heads)
            mask_contexts.append(ctx)
        for ctx in mask_contexts:
            ctx.__enter__()
        random_accs.append(validate_accuracy(model, val_loader, device))
        for ctx in reversed(mask_contexts):
            ctx.__exit__(None, None, None)

    mask_random_acc = float(np.mean(random_accs))

    return {
        "no_mask": no_mask_acc,
        "mask_local": mask_local_acc,
        "mask_global": mask_global_acc,
        "mask_random": mask_random_acc,
        "mask_random_trials": random_accs,
        "head_roles": {str(k): v for k, v in head_roles.items()},
    }


def evaluate_model(model_name, ckpt_path, val_loader, device, dist_matrix, num_batches):
    """Full evaluation pipeline for one model."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    model = load_deit_small(pretrained=False, device=device)
    model = load_checkpoint(model, ckpt_path, device)

    print("Computing metrics...")
    metrics = compute_all_metrics(model, val_loader, device, dist_matrix, num_batches)

    print("Running GMM bimodality test...")
    gmm_results = gmm_bimodality_test(model, val_loader, device, dist_matrix, num_batches)

    print("Checking role persistence...")
    persistence = role_persistence(model, val_loader, device, dist_matrix, num_batches)

    print("Running head masking experiment...")
    masking = head_masking_experiment(model, val_loader, device, dist_matrix)

    return {
        "metrics": {str(b): {k: v.tolist() for k, v in bv.items()} for b, bv in metrics.items()},
        "gmm": {str(b): v for b, v in gmm_results.items()},
        "persistence": {str(b): v for b, v in persistence.items()},
        "masking": masking,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_ckpt", type=str, required=True, help="Path to baseline best.pth")
    parser.add_argument("--regularized_ckpt", type=str, required=True, help="Path to regularized best.pth")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_batches", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    device = cfg.DEVICE
    output_dir = args.output_dir or cfg.ANALYSIS_DIR
    os.makedirs(output_dir, exist_ok=True)
    num_batches = args.num_batches or cfg.NUM_EVAL_BATCHES

    val_loader = get_val_loader(data_root=args.data_root, batch_size=args.batch_size)
    dist_matrix = build_distance_matrix(device=device)

    baseline_results = evaluate_model(
        "Baseline FT", args.baseline_ckpt, val_loader, device, dist_matrix, num_batches
    )
    regularized_results = evaluate_model(
        "Regularized FT", args.regularized_ckpt, val_loader, device, dist_matrix, num_batches
    )

    combined = {
        "baseline": baseline_results,
        "regularized": regularized_results,
    }

    output_path = os.path.join(output_dir, "evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for name, res in [("Baseline", baseline_results), ("Regularized", regularized_results)]:
        mask = res["masking"]
        print(f"\n{name}:")
        print(f"  Val Acc (no mask): {mask['no_mask']:.2f}%")
        print(f"  Val Acc (mask local): {mask['mask_local']:.2f}%  (drop: {mask['no_mask'] - mask['mask_local']:+.2f})")
        print(f"  Val Acc (mask global): {mask['mask_global']:.2f}%  (drop: {mask['no_mask'] - mask['mask_global']:+.2f})")
        print(f"  Val Acc (mask random): {mask['mask_random']:.2f}%  (drop: {mask['no_mask'] - mask['mask_random']:+.2f})")

    print("\nGMM Bimodality (BIC diff, positive = 2-comp better):")
    for bidx in cfg.REGULARIZED_BLOCKS:
        b_bic = baseline_results["gmm"][str(bidx)]["bic_diff"]
        r_bic = regularized_results["gmm"][str(bidx)]["bic_diff"]
        print(f"  Block {bidx}: baseline={b_bic:+.1f}  regularized={r_bic:+.1f}")

    print("\nRole Persistence (mean across heads):")
    for bidx in cfg.REGULARIZED_BLOCKS:
        b_p = baseline_results["persistence"][str(bidx)]["mean_persistence"]
        r_p = regularized_results["persistence"][str(bidx)]["mean_persistence"]
        print(f"  Block {bidx}: baseline={b_p:.3f}  regularized={r_p:.3f}")


if __name__ == "__main__":
    main()
