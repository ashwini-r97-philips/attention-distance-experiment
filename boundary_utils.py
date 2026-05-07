"""Boundary extraction and boundary-conditional attention metrics."""

import numpy as np
import torch
import torch.nn.functional as F


def extract_boundary_mask(seg_mask, thickness=2):
    """Extract boundary pixels from a segmentation mask using morphological gradient.

    Args:
        seg_mask: (H, W) integer tensor or ndarray of class labels.
        thickness: boundary width in pixels.

    Returns:
        boundary: (H, W) boolean array — True at boundary pixels.
    """
    if isinstance(seg_mask, torch.Tensor):
        seg_mask = seg_mask.cpu().numpy()
    seg_mask = seg_mask.astype(np.int32)

    # A pixel is on a boundary if any neighbor within `thickness` has a different label
    from scipy.ndimage import maximum_filter, minimum_filter
    size = 2 * thickness + 1
    local_max = maximum_filter(seg_mask, size=size)
    local_min = minimum_filter(seg_mask, size=size)
    boundary = local_max != local_min
    return boundary


def get_boundary_token_mask(seg_mask, grid_h, grid_w, thickness=2, ignore_index=255):
    """Determine which patch tokens overlap with boundaries.

    Args:
        seg_mask: (H, W) integer tensor of class labels.
        grid_h, grid_w: patch grid dimensions (e.g. 32, 32).
        thickness: boundary width.
        ignore_index: label to ignore.

    Returns:
        token_boundary_mask: (grid_h * grid_w,) boolean tensor.
            True if the patch contains boundary pixels.
    """
    if isinstance(seg_mask, torch.Tensor):
        seg_mask_np = seg_mask.cpu().numpy()
    else:
        seg_mask_np = seg_mask

    H, W = seg_mask_np.shape
    boundary = extract_boundary_mask(seg_mask_np, thickness=thickness)

    # Downsample boundary map to patch grid by checking if any boundary pixel
    # falls within each patch
    patch_h = H // grid_h
    patch_w = W // grid_w

    token_mask = np.zeros(grid_h * grid_w, dtype=bool)
    for r in range(grid_h):
        for c in range(grid_w):
            patch = boundary[r * patch_h:(r + 1) * patch_h, c * patch_w:(c + 1) * patch_w]
            token_mask[r * grid_w + c] = patch.any()

    return torch.tensor(token_mask, dtype=torch.bool)


def compute_conditional_mad(attn_weights, dist_matrix, boundary_token_mask, exclude_cls=True):
    """Compute MAD separately for boundary tokens vs interior tokens.

    Args:
        attn_weights: (B, H, N, N) attention weights. N includes CLS.
        dist_matrix: (N_patches, N_patches) normalized distances.
        boundary_token_mask: (N_patches,) boolean — True for boundary tokens.
        exclude_cls: remove CLS token.

    Returns:
        boundary_mad: (H,) — MAD averaged over boundary query tokens.
        interior_mad: (H,) — MAD averaged over interior query tokens.
    """
    if exclude_cls:
        attn_weights = attn_weights[:, :, 1:, 1:]

    B, H, N, N2 = attn_weights.shape
    R = dist_matrix[:N, :N2].to(attn_weights.device)
    mask = boundary_token_mask[:N].to(attn_weights.device)

    weighted = (attn_weights * R.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (B, H, N)

    # Boundary tokens
    if mask.any():
        boundary_mad = weighted[:, :, mask].mean(dim=(0, 2))
    else:
        boundary_mad = torch.zeros(H, device=attn_weights.device)

    # Interior tokens
    interior_mask = ~mask
    if interior_mask.any():
        interior_mad = weighted[:, :, interior_mask].mean(dim=(0, 2))
    else:
        interior_mad = torch.zeros(H, device=attn_weights.device)

    return boundary_mad, interior_mad


def compute_boundary_f1(pred, target, thickness=2, ignore_index=255):
    """Compute boundary F1 score between predicted and ground truth segmentation.

    Args:
        pred: (H, W) predicted class labels.
        target: (H, W) ground truth class labels.
        thickness: boundary width.
        ignore_index: label to ignore.

    Returns:
        f1: float, boundary F1 score.
        precision: float.
        recall: float.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    # Ignore unlabeled
    valid = target != ignore_index
    pred_boundary = extract_boundary_mask(pred, thickness=thickness)
    gt_boundary = extract_boundary_mask(target, thickness=thickness)

    # Only evaluate in valid regions
    pred_boundary = pred_boundary & valid
    gt_boundary = gt_boundary & valid

    tp = (pred_boundary & gt_boundary).sum()
    fp = (pred_boundary & ~gt_boundary).sum()
    fn = (~pred_boundary & gt_boundary).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return float(f1), float(precision), float(recall)


def compute_miou(pred, target, num_classes, ignore_index=255):
    """Compute mean Intersection over Union.

    Args:
        pred: (N,) or (H, W) predicted class labels.
        target: (N,) or (H, W) ground truth labels.
        num_classes: number of classes.
        ignore_index: label to ignore.

    Returns:
        miou: float.
        per_class_iou: dict {class_idx: iou}.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    pred = pred.flatten()
    target = target.flatten()

    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    per_class_iou = {}
    ious = []
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        intersection = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()
        if union > 0:
            iou = intersection / union
            ious.append(iou)
            per_class_iou[c] = float(iou)

    miou = float(np.mean(ious)) if ious else 0.0
    return miou, per_class_iou
