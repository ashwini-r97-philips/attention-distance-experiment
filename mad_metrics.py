"""Core attention-distance metrics: MAD, non-self MAD, local mass, entropy."""

import torch
import config as cfg


def build_distance_matrix(grid_h=None, grid_w=None, device="cpu"):
    """Build normalized pairwise L2 distance matrix for a 2D patch grid.

    Returns:
        R: tensor of shape (grid_h*grid_w, grid_h*grid_w) with distances
           normalized to [0, 1] by dividing by the maximum distance.
    """
    grid_h = grid_h or cfg.GRID_H
    grid_w = grid_w or cfg.GRID_W
    n = grid_h * grid_w

    # (n, 2) coordinates
    rows = torch.arange(grid_h, device=device).float()
    cols = torch.arange(grid_w, device=device).float()
    grid = torch.stack(torch.meshgrid(rows, cols, indexing="ij"), dim=-1)  # (H, W, 2)
    coords = grid.reshape(n, 2)  # (N_patches, 2)

    # Pairwise L2
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # (N, N, 2)
    R = diff.norm(dim=-1)  # (N, N)
    R = R / R.max()  # normalize to [0, 1]
    return R


def compute_mad(attn_weights, dist_matrix, exclude_cls=True):
    """Mean Attention Distance per head.

    Args:
        attn_weights: (B, H, N, N) post-softmax attention. N includes CLS token.
        dist_matrix: (N_patches, N_patches) normalized distances.
        exclude_cls: if True, remove the first token (CLS) from both query and key.

    Returns:
        (H,) tensor — MAD averaged over batch and query tokens.
    """
    if exclude_cls:
        # Remove CLS row and column (position 0)
        attn_weights = attn_weights[:, :, 1:, 1:]  # (B, H, N_p, N_p)

    B, H, N, N2 = attn_weights.shape
    R = dist_matrix[:N, :N2].to(attn_weights.device)  # (N, N2)

    # MAD = mean_over_{b,i} sum_j A_{b,h,i,j} * R_{i,j}
    # (B, H, N, N2) * (N, N2) -> sum over j -> (B, H, N) -> mean over B, N -> (H,)
    weighted = (attn_weights * R.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (B, H, N)
    mad = weighted.mean(dim=(0, 2))  # (H,)
    return mad


def compute_non_self_mad(attn_weights, dist_matrix, exclude_cls=True):
    """MAD excluding self-attention (diagonal)."""
    if exclude_cls:
        attn_weights = attn_weights[:, :, 1:, 1:]

    B, H, N, N2 = attn_weights.shape
    R = dist_matrix[:N, :N2].to(attn_weights.device)

    # Mask out diagonal
    mask = ~torch.eye(N, N2, dtype=torch.bool, device=attn_weights.device)
    # Re-normalize attention after removing diagonal
    attn_masked = attn_weights * mask.unsqueeze(0).unsqueeze(0)
    attn_sum = attn_masked.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    attn_renorm = attn_masked / attn_sum

    weighted = (attn_renorm * R.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (B, H, N)
    mad = weighted.mean(dim=(0, 2))  # (H,)
    return mad


def compute_local_mass(attn_weights, dist_matrix, tau=None, exclude_cls=True, grid_h=None, grid_w=None):
    """Fraction of attention within radius tau (in normalized distance)."""
    tau = tau if tau is not None else cfg.LOCAL_RADIUS_TAU
    if exclude_cls:
        attn_weights = attn_weights[:, :, 1:, 1:]

    B, H, N, N2 = attn_weights.shape
    R = dist_matrix[:N, :N2].to(attn_weights.device)

    # Convert tau from patch units to normalized units
    gh = grid_h or cfg.GRID_H
    gw = grid_w or cfg.GRID_W
    max_dist_patches = ((gh - 1)**2 + (gw - 1)**2) ** 0.5
    tau_normalized = tau / max_dist_patches

    local_mask = (R <= tau_normalized).float()  # (N, N2)
    local_attn = (attn_weights * local_mask.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (B, H, N)
    mass = local_attn.mean(dim=(0, 2))  # (H,)
    return mass


def compute_attention_entropy(attn_weights, exclude_cls=True):
    """Shannon entropy of attention distribution per head."""
    if exclude_cls:
        attn_weights = attn_weights[:, :, 1:, 1:]

    # Clamp to avoid log(0)
    eps = 1e-8
    attn_clamped = attn_weights.clamp(min=eps)
    entropy = -(attn_clamped * attn_clamped.log()).sum(dim=-1)  # (B, H, N)
    return entropy.mean(dim=(0, 2))  # (H,)


def compute_distance_histogram(attn_weights, dist_matrix, num_bins=10, exclude_cls=True):
    """Histogram of attention mass by distance quantile per head.

    Returns:
        (H, num_bins) tensor — fraction of total attention in each distance bin.
    """
    if exclude_cls:
        attn_weights = attn_weights[:, :, 1:, 1:]

    B, H, N, N2 = attn_weights.shape
    R = dist_matrix[:N, :N2].to(attn_weights.device)

    # Compute distance bin edges (uniform quantiles)
    r_flat = R.flatten()
    quantiles = torch.linspace(0, 1, num_bins + 1, device=R.device)
    edges = torch.quantile(r_flat, quantiles)
    edges[-1] = edges[-1] + 1e-6  # ensure last bin captures max

    hist = torch.zeros(H, num_bins, device=attn_weights.device)
    for i in range(num_bins):
        mask = ((R >= edges[i]) & (R < edges[i + 1])).float()  # (N, N2)
        bin_attn = (attn_weights * mask.unsqueeze(0).unsqueeze(0)).sum(dim=(0, 2, 3))  # (H,)
        hist[:, i] = bin_attn

    # Normalize so each head sums to 1
    hist = hist / hist.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return hist
