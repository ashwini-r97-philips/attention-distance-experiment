"""Attention-distance metrics: MAD, local mass, entropy, inter-head variance."""

import torch


def build_distance_matrix(grid_h, grid_w, device="cpu"):
    """Build normalised pairwise L2 distance matrix for a 2D patch grid.

    Returns:
        R: (grid_h*grid_w, grid_h*grid_w) with distances in [0, 1].
    """
    n = grid_h * grid_w
    rows = torch.arange(grid_h, device=device).float()
    cols = torch.arange(grid_w, device=device).float()
    grid = torch.stack(torch.meshgrid(rows, cols, indexing="ij"), dim=-1)
    coords = grid.reshape(n, 2)
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)
    R = diff.norm(dim=-1)
    R = R / R.max()
    return R


def _slice_patch_only(attn_weights, exclude_cls):
    """Remove CLS token from attention if requested."""
    if exclude_cls:
        return attn_weights[:, :, 1:, 1:]
    return attn_weights


def compute_mad(attn_weights, dist_matrix, exclude_cls=True):
    """Mean Attention Distance per head.

    Args:
        attn_weights: (B, H, N, N) post-softmax attention.
        dist_matrix: (N_patches, N_patches) normalised distances.
        exclude_cls: if True, remove CLS token from both query and key.

    Returns:
        (H,) tensor — MAD averaged over batch and query tokens.
    """
    attn_weights = _slice_patch_only(attn_weights, exclude_cls)
    B, H, N, N2 = attn_weights.shape
    R = dist_matrix[:N, :N2].to(attn_weights.device)
    weighted = (attn_weights * R.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
    return weighted.mean(dim=(0, 2))


def compute_local_mass(attn_weights, dist_matrix, tau, exclude_cls=True):
    """Fraction of attention within normalised distance tau.

    Args:
        tau: float in [0, 1] (already normalised).
    """
    attn_weights = _slice_patch_only(attn_weights, exclude_cls)
    B, H, N, N2 = attn_weights.shape
    R = dist_matrix[:N, :N2].to(attn_weights.device)
    local_mask = (R <= tau).float()
    local_attn = (attn_weights * local_mask.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
    return local_attn.mean(dim=(0, 2))


def compute_attention_entropy(attn_weights, exclude_cls=True):
    """Shannon entropy of attention distribution per head."""
    attn_weights = _slice_patch_only(attn_weights, exclude_cls)
    eps = 1e-8
    a = attn_weights.clamp(min=eps)
    entropy = -(a * a.log()).sum(dim=-1)
    return entropy.mean(dim=(0, 2))


def compute_inter_head_mad_variance(head_mads):
    """Variance of MAD across heads within a single layer.

    Args:
        head_mads: (H,) tensor of per-head MAD values.
    Returns:
        scalar tensor.
    """
    return head_mads.var()


def compute_distance_histogram(attn_weights, dist_matrix, num_bins=10, exclude_cls=True):
    """Histogram of attention mass by distance bin per head.

    Returns:
        (H, num_bins) tensor — fraction of total attention in each bin.
    """
    attn_weights = _slice_patch_only(attn_weights, exclude_cls)
    B, H, N, N2 = attn_weights.shape
    R = dist_matrix[:N, :N2].to(attn_weights.device)

    r_flat = R.flatten()
    quantiles = torch.linspace(0, 1, num_bins + 1, device=R.device)
    edges = torch.quantile(r_flat, quantiles)
    edges[-1] = edges[-1] + 1e-6

    hist = torch.zeros(H, num_bins, device=attn_weights.device)
    for i in range(num_bins):
        mask = ((R >= edges[i]) & (R < edges[i + 1])).float()
        bin_attn = (attn_weights * mask.unsqueeze(0).unsqueeze(0)).sum(dim=(0, 2, 3))
        hist[:, i] = bin_attn

    hist = hist / hist.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return hist


def compute_head_correlation(attn_weights, exclude_cls=True):
    """Pairwise correlation between heads' flattened attention maps within a layer.

    Args:
        attn_weights: (B, H, N, N)
    Returns:
        (H, H) correlation matrix, averaged over batch.
    """
    attn_weights = _slice_patch_only(attn_weights, exclude_cls)
    B, H, N, N2 = attn_weights.shape
    flat = attn_weights.reshape(B, H, -1)  # (B, H, N*N2)
    # Subtract mean per head
    flat = flat - flat.mean(dim=-1, keepdim=True)
    # Normalise
    norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    flat = flat / norms
    # Correlation: (B, H, H)
    corr = torch.bmm(flat, flat.transpose(1, 2))
    return corr.mean(dim=0)
