"""Segmentation evaluation metrics: mIoU and boundary F1.

Self-contained implementation; no dependency on bimodal_head_specialisation.
"""

import torch
import numpy as np
from scipy.ndimage import binary_dilation


def compute_miou(pred_flat, target_flat, num_classes, ignore_index=255):
    """Mean IoU over all classes present in targets.

    Args:
        pred_flat:   (N,) int64 predicted class labels
        target_flat: (N,) int64 ground-truth labels (ignore_index pixels excluded)
        num_classes: number of semantic classes
        ignore_index: label value to ignore

    Returns:
        miou: float
        per_class_iou: list of float, length num_classes (NaN for absent classes)
    """
    valid = target_flat != ignore_index
    pred_flat = pred_flat[valid]
    target_flat = target_flat[valid]

    per_class = []
    for c in range(num_classes):
        pred_c = pred_flat == c
        gt_c = target_flat == c
        intersection = (pred_c & gt_c).sum().item()
        union = (pred_c | gt_c).sum().item()
        if union == 0:
            per_class.append(float("nan"))
        else:
            per_class.append(intersection / union)

    valid_iou = [v for v in per_class if not np.isnan(v)]
    miou = float(np.mean(valid_iou)) if valid_iou else 0.0
    return miou, per_class


def compute_boundary_f1(pred, target, ignore_index=255, thickness=2):
    """Boundary F1 score between predicted and ground-truth segmentation boundaries.

    Args:
        pred:   (H, W) int64 tensor — predicted labels
        target: (H, W) int64 tensor — ground-truth labels
        ignore_index: pixels to exclude
        thickness: dilation radius for boundary matching tolerance

    Returns:
        f1: float
        precision: float
        recall: float
    """
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    valid = target_np != ignore_index

    def get_boundary(seg, valid_mask):
        boundary = np.zeros_like(seg, dtype=bool)
        for shift in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            shifted = np.roll(seg, shift, axis=(0, 1))
            boundary |= (seg != shifted)
        return boundary & valid_mask

    pred_boundary = get_boundary(pred_np, valid)
    gt_boundary = get_boundary(target_np, valid)

    struct = np.ones((thickness * 2 + 1, thickness * 2 + 1), dtype=bool)
    pred_dilated = binary_dilation(pred_boundary, structure=struct)
    gt_dilated = binary_dilation(gt_boundary, structure=struct)

    tp_pred = (pred_boundary & gt_dilated).sum()
    tp_gt = (gt_boundary & pred_dilated).sum()

    precision = tp_pred / max(pred_boundary.sum(), 1)
    recall = tp_gt / max(gt_boundary.sum(), 1)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return float(f1), float(precision), float(recall)
