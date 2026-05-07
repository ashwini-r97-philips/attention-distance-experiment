"""Post-training evaluation for segmentation: mIoU, boundary F1, conditional MAD,
GMM bimodality, role persistence, head masking with dense prediction metrics."""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

import seg_config as cfg
from seg_model import DeiTSegModel
from attention_hooks import capture_attention, patch_attention_forward, unpatch_attention_forward
from model_utils import mask_heads
from mad_metrics import (
    build_distance_matrix,
    compute_mad,
    compute_non_self_mad,
    compute_local_mass,
    compute_attention_entropy,
    compute_distance_histogram,
)
from boundary_utils import (
    compute_miou,
    compute_boundary_f1,
    compute_conditional_mad,
    get_boundary_token_mask,
)
from seg_data import get_val_loader


def load_seg_checkpoint(ckpt_path, device):
    model = DeiTSegModel(pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_all_metrics(model, val_loader, device, dist_matrix, num_batches):
    """Compute MAD, non-self MAD, local mass, entropy, distance histograms per block."""
    model.eval()
    all_blocks = list(range(cfg.NUM_BLOCKS))
    accum = {b: {"mad": [], "ns_mad": [], "lm": [], "ent": [], "dist_hist": []} for b in all_blocks}

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(val_loader, total=num_batches, desc="Metrics")):
            if batch_idx >= num_batches:
                break
            images = images.to(device)
            with capture_attention(model.encoder, all_blocks) as get_attn:
                _ = model(images, return_aux=False)
                attn_dict = get_attn()
            for bidx in all_blocks:
                attn = attn_dict[bidx]
                accum[bidx]["mad"].append(compute_mad(attn, dist_matrix).cpu().numpy())
                accum[bidx]["ns_mad"].append(compute_non_self_mad(attn, dist_matrix).cpu().numpy())
                accum[bidx]["lm"].append(compute_local_mass(attn, dist_matrix, tau=cfg.LOCAL_RADIUS_TAU).cpu().numpy())
                accum[bidx]["ent"].append(compute_attention_entropy(attn).cpu().numpy())
                accum[bidx]["dist_hist"].append(compute_distance_histogram(attn, dist_matrix).cpu().numpy())

    result = {}
    for b in all_blocks:
        result[b] = {
            "mad": np.mean(accum[b]["mad"], axis=0),
            "non_self_mad": np.mean(accum[b]["ns_mad"], axis=0),
            "local_mass": np.mean(accum[b]["lm"], axis=0),
            "entropy": np.mean(accum[b]["ent"], axis=0),
            "dist_hist": np.mean(accum[b]["dist_hist"], axis=0),  # (H, num_bins)
        }
    return result


# ─── GMM Bimodality ─────────────────────────────────────────────────────────

def gmm_bimodality_test(model, val_loader, device, dist_matrix, num_batches):
    model.eval()
    early_blocks = cfg.REGULARIZED_BLOCKS
    mad_per_batch = {b: [] for b in early_blocks}

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(val_loader, total=num_batches, desc="GMM")):
            if batch_idx >= num_batches:
                break
            images = images.to(device)
            with capture_attention(model.encoder, early_blocks) as get_attn:
                _ = model(images, return_aux=False)
                attn_dict = get_attn()
            for bidx in early_blocks:
                mad_per_batch[bidx].append(compute_mad(attn_dict[bidx], dist_matrix).cpu().numpy())

    results = {}
    for bidx in early_blocks:
        all_mads = np.stack(mad_per_batch[bidx], axis=0)  # (num_batches, num_heads)
        flat = all_mads.flatten().reshape(-1, 1)
        gmm1 = GaussianMixture(n_components=1, random_state=42).fit(flat)
        gmm2 = GaussianMixture(n_components=2, random_state=42).fit(flat)
        results[bidx] = {
            "bic_1comp": gmm1.bic(flat),
            "bic_2comp": gmm2.bic(flat),
            "bic_diff": gmm1.bic(flat) - gmm2.bic(flat),
            "gmm2_means": gmm2.means_.flatten().tolist(),
            "gmm2_weights": gmm2.weights_.tolist(),
            "per_head_mean_mad": all_mads.mean(axis=0).tolist(),
        }
    return results


# ─── Role Persistence ────────────────────────────────────────────────────────

def role_persistence(model, val_loader, device, dist_matrix, num_batches):
    model.eval()
    early_blocks = cfg.REGULARIZED_BLOCKS
    assignments = {b: [] for b in early_blocks}

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(val_loader, total=num_batches, desc="Persistence")):
            if batch_idx >= num_batches:
                break
            images = images.to(device)
            with capture_attention(model.encoder, early_blocks) as get_attn:
                _ = model(images, return_aux=False)
                attn_dict = get_attn()
            for bidx in early_blocks:
                mads = compute_mad(attn_dict[bidx], dist_matrix).cpu().numpy()
                median = np.median(mads)
                assignments[bidx].append((mads > median).astype(int))

    results = {}
    for bidx in early_blocks:
        all_a = np.stack(assignments[bidx], axis=0)
        persistence_per_head = []
        for h in range(cfg.NUM_HEADS):
            counts = np.bincount(all_a[:, h], minlength=2)
            persistence_per_head.append(counts.max() / counts.sum())
        results[bidx] = {
            "persistence_per_head": persistence_per_head,
            "mean_persistence": float(np.mean(persistence_per_head)),
            "dominant_role": [int(np.argmax(np.bincount(all_a[:, h], minlength=2))) for h in range(cfg.NUM_HEADS)],
        }
    return results


# ─── Boundary-Conditional MAD ────────────────────────────────────────────────

def conditional_mad_analysis(model, val_loader, device, dist_matrix, num_batches):
    """Compute MAD separately for boundary vs interior tokens."""
    model.eval()
    early_blocks = cfg.REGULARIZED_BLOCKS
    boundary_mads = {b: [] for b in early_blocks}
    interior_mads = {b: [] for b in early_blocks}

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(val_loader, total=num_batches, desc="Conditional MAD")):
            if batch_idx >= num_batches:
                break
            images = images.to(device)

            with capture_attention(model.encoder, early_blocks) as get_attn:
                _ = model(images, return_aux=False)
                attn_dict = get_attn()

            # For each image in batch, compute boundary token mask
            for b_img in range(masks.shape[0]):
                bt_mask = get_boundary_token_mask(
                    masks[b_img], cfg.GRID_H, cfg.GRID_W,
                    thickness=cfg.BOUNDARY_THICKNESS, ignore_index=cfg.IGNORE_INDEX
                )
                for bidx in early_blocks:
                    # Single image attention
                    attn_single = attn_dict[bidx][b_img:b_img + 1]
                    b_mad, i_mad = compute_conditional_mad(attn_single, dist_matrix, bt_mask)
                    boundary_mads[bidx].append(b_mad.cpu().numpy())
                    interior_mads[bidx].append(i_mad.cpu().numpy())

    results = {}
    for bidx in early_blocks:
        results[bidx] = {
            "boundary_mad": np.mean(boundary_mads[bidx], axis=0).tolist(),
            "interior_mad": np.mean(interior_mads[bidx], axis=0).tolist(),
        }
    return results


# ─── Head Masking → mIoU + boundary F1 ──────────────────────────────────────

@torch.no_grad()
def validate_seg(model, val_loader, device):
    """Compute mIoU and boundary F1."""
    model.eval()
    all_pred, all_target = [], []
    bf1s = []
    for images, masks in val_loader:
        images = images.to(device)
        logits = model(images, return_aux=False)
        pred = logits.argmax(dim=1)
        for b in range(pred.shape[0]):
            all_pred.append(pred[b].cpu())
            all_target.append(masks[b])
            f1, _, _ = compute_boundary_f1(pred[b], masks[b], ignore_index=cfg.IGNORE_INDEX)
            bf1s.append(f1)

    pred_cat = torch.cat([p.flatten() for p in all_pred])
    target_cat = torch.cat([t.flatten() for t in all_target])
    miou, _ = compute_miou(pred_cat, target_cat, cfg.NUM_SEG_CLASSES, ignore_index=cfg.IGNORE_INDEX)
    return miou, float(np.mean(bf1s))


def head_masking_experiment(model, val_loader, device, dist_matrix, num_batches=20):
    """Mask local/global/random heads, measure mIoU and boundary F1 impact."""
    model.eval()
    early_blocks = cfg.REGULARIZED_BLOCKS

    # Determine head roles from val MADs
    mad_accum = {b: [] for b in early_blocks}
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(val_loader):
            if batch_idx >= num_batches:
                break
            images = images.to(device)
            with capture_attention(model.encoder, early_blocks) as get_attn:
                _ = model(images, return_aux=False)
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

    # No mask
    no_mask_miou, no_mask_bf1 = validate_seg(model, val_loader, device)

    # Mask local heads
    ctxs = [mask_heads(model.encoder, bidx, head_roles[bidx]["local_heads"]) for bidx in early_blocks]
    for c in ctxs:
        c.__enter__()
    mask_local_miou, mask_local_bf1 = validate_seg(model, val_loader, device)
    for c in reversed(ctxs):
        c.__exit__(None, None, None)

    # Mask global heads
    ctxs = [mask_heads(model.encoder, bidx, head_roles[bidx]["global_heads"]) for bidx in early_blocks]
    for c in ctxs:
        c.__enter__()
    mask_global_miou, mask_global_bf1 = validate_seg(model, val_loader, device)
    for c in reversed(ctxs):
        c.__exit__(None, None, None)

    # Mask random heads
    rng = np.random.RandomState(42)
    random_mious, random_bf1s = [], []
    for _ in range(cfg.NUM_MASK_TRIALS):
        ctxs = []
        for bidx in early_blocks:
            rh = rng.choice(cfg.NUM_HEADS, size=cfg.NUM_LOW_HEADS, replace=False).tolist()
            ctxs.append(mask_heads(model.encoder, bidx, rh))
        for c in ctxs:
            c.__enter__()
        m, b = validate_seg(model, val_loader, device)
        random_mious.append(m)
        random_bf1s.append(b)
        for c in reversed(ctxs):
            c.__exit__(None, None, None)

    return {
        "no_mask": {"miou": no_mask_miou, "boundary_f1": no_mask_bf1},
        "mask_local": {"miou": mask_local_miou, "boundary_f1": mask_local_bf1},
        "mask_global": {"miou": mask_global_miou, "boundary_f1": mask_global_bf1},
        "mask_random": {"miou": float(np.mean(random_mious)), "boundary_f1": float(np.mean(random_bf1s))},
        "head_roles": {str(k): v for k, v in head_roles.items()},
    }


# ─── Head Output Norms ──────────────────────────────────────────────────────

@torch.no_grad()
def compute_head_output_norms(model, val_loader, device, num_batches=10):
    """Compute per-head output norm to verify heads aren't dead."""
    model.eval()
    early_blocks = cfg.REGULARIZED_BLOCKS
    norms = {b: [] for b in early_blocks}

    for batch_idx, (images, _) in enumerate(val_loader):
        if batch_idx >= num_batches:
            break
        images = images.to(device)

        # Forward through encoder manually, intercept block outputs
        x = model.encoder.patch_embed(images)
        x = model.encoder._pos_embed(x)
        x = model.encoder.patch_drop(x)
        x = model.encoder.norm_pre(x)

        for i, blk in enumerate(model.encoder.blocks):
            x_pre = x
            x = blk(x)
            if i in early_blocks:
                # The attention output contribution is x - x_pre (roughly, before MLP)
                # More precisely, get the attention sublayer output
                attn_out = blk.attn(blk.norm1(x_pre))
                B, N, C = attn_out.shape
                head_dim = C // cfg.NUM_HEADS
                per_head = attn_out.reshape(B, N, cfg.NUM_HEADS, head_dim)
                head_norms = per_head.norm(dim=-1).mean(dim=(0, 1))  # (H,)
                norms[i].append(head_norms.cpu().numpy())

    result = {}
    for bidx in early_blocks:
        result[bidx] = np.mean(norms[bidx], axis=0).tolist()
    return result


# ─── Main ────────────────────────────────────────────────────────────────────

def evaluate_model(name, ckpt_path, val_loader, device, dist_matrix, num_batches):
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {name}")
    print(f"{'=' * 60}")

    model = load_seg_checkpoint(ckpt_path, device)

    print("Computing all metrics...")
    metrics = compute_all_metrics(model, val_loader, device, dist_matrix, num_batches)

    print("GMM bimodality test...")
    gmm = gmm_bimodality_test(model, val_loader, device, dist_matrix, num_batches)

    print("Role persistence...")
    persistence = role_persistence(model, val_loader, device, dist_matrix, num_batches)

    print("Conditional MAD (boundary vs interior)...")
    cond_mad = conditional_mad_analysis(model, val_loader, device, dist_matrix, min(num_batches, 30))

    print("Head masking experiment...")
    masking = head_masking_experiment(model, val_loader, device, dist_matrix)

    print("Head output norms...")
    head_norms = compute_head_output_norms(model, val_loader, device)

    def _serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    return {
        "metrics": {str(b): {k: _serialize(v) for k, v in bv.items()} for b, bv in metrics.items()},
        "gmm": {str(b): v for b, v in gmm.items()},
        "persistence": {str(b): v for b, v in persistence.items()},
        "conditional_mad": {str(b): v for b, v in cond_mad.items()},
        "masking": masking,
        "head_norms": {str(b): v for b, v in head_norms.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_ckpt", type=str, required=True)
    parser.add_argument("--regularized_ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    device = cfg.DEVICE
    output_dir = args.output_dir or cfg.ANALYSIS_DIR
    os.makedirs(output_dir, exist_ok=True)
    num_batches = args.num_batches or cfg.NUM_EVAL_BATCHES

    val_loader = get_val_loader(batch_size=args.batch_size)
    dist_matrix = build_distance_matrix(grid_h=cfg.GRID_H, grid_w=cfg.GRID_W, device=device)

    baseline_results = evaluate_model("Baseline Seg", args.baseline_ckpt, val_loader, device, dist_matrix, num_batches)
    regularized_results = evaluate_model("Regularized Seg", args.regularized_ckpt, val_loader, device, dist_matrix, num_batches)

    combined = {"baseline": baseline_results, "regularized": regularized_results}
    output_path = os.path.join(output_dir, "seg_evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    for name, res in [("Baseline", baseline_results), ("Regularized", regularized_results)]:
        m = res["masking"]
        print(f"\n{name}:")
        print(f"  mIoU (no mask): {m['no_mask']['miou']:.4f}  bF1: {m['no_mask']['boundary_f1']:.4f}")
        print(f"  mIoU (mask local):  {m['mask_local']['miou']:.4f}  bF1: {m['mask_local']['boundary_f1']:.4f}")
        print(f"  mIoU (mask global): {m['mask_global']['miou']:.4f}  bF1: {m['mask_global']['boundary_f1']:.4f}")
        print(f"  mIoU (mask random): {m['mask_random']['miou']:.4f}  bF1: {m['mask_random']['boundary_f1']:.4f}")

    print("\nConditional MAD (boundary vs interior tokens):")
    for bidx in cfg.REGULARIZED_BLOCKS:
        bc = baseline_results["conditional_mad"][str(bidx)]
        rc = regularized_results["conditional_mad"][str(bidx)]
        print(f"  Block {bidx}: Baseline boundary={[f'{v:.4f}' for v in bc['boundary_mad']]}  "
              f"interior={[f'{v:.4f}' for v in bc['interior_mad']]}")
        print(f"           Regularized boundary={[f'{v:.4f}' for v in rc['boundary_mad']]}  "
              f"interior={[f'{v:.4f}' for v in rc['interior_mad']]}")

    print("\nGMM Bimodality (BIC diff, positive = 2-comp better):")
    for bidx in cfg.REGULARIZED_BLOCKS:
        b_bic = baseline_results["gmm"][str(bidx)]["bic_diff"]
        r_bic = regularized_results["gmm"][str(bidx)]["bic_diff"]
        print(f"  Block {bidx}: baseline={b_bic:+.1f}  regularized={r_bic:+.1f}")

    print("\nHead Output Norms:")
    for bidx in cfg.REGULARIZED_BLOCKS:
        bn = baseline_results["head_norms"][str(bidx)]
        rn = regularized_results["head_norms"][str(bidx)]
        print(f"  Block {bidx}: baseline={[f'{v:.3f}' for v in bn]}  regularized={[f'{v:.3f}' for v in rn]}")


if __name__ == "__main__":
    main()
