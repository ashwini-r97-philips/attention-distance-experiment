"""Bimodal head specialization loss: gap + compactness."""

import torch
import config as cfg
from mad_metrics import compute_mad, build_distance_matrix


class BimodalHeadLoss(torch.nn.Module):
    """Encourages bimodal distribution of headwise MAD in early blocks.

    For each regularized block:
      - Sort heads by MAD (detached for index selection)
      - Bottom half = "local" group, top half = "global" group
      - L_gap:     push the two groups apart (at least delta apart)
      - L_compact: make each group internally tight
    """

    def __init__(
        self,
        delta=None,
        lambda_gap=None,
        lambda_compact=None,
        num_low=None,
        num_high=None,
    ):
        super().__init__()
        self.delta = delta if delta is not None else cfg.DELTA
        self.lambda_gap = lambda_gap if lambda_gap is not None else cfg.LAMBDA_GAP
        self.lambda_compact = lambda_compact if lambda_compact is not None else cfg.LAMBDA_COMPACT
        self.num_low = num_low or cfg.NUM_LOW_HEADS
        self.num_high = num_high or cfg.NUM_HIGH_HEADS
        self._dist_matrix = None

    def _get_dist_matrix(self, device):
        if self._dist_matrix is None or self._dist_matrix.device != device:
            self._dist_matrix = build_distance_matrix(device=device)
        return self._dist_matrix

    def forward(self, attn_dict, warmup_factor=1.0):
        """Compute the bimodal regularization loss.

        Args:
            attn_dict: {block_idx: attn_weights (B, H, N, N)} with gradients.
            warmup_factor: float in [0, 1], linearly ramps the loss.

        Returns:
            loss: scalar tensor
            info: dict with per-block gap and compactness values for logging
        """
        if not attn_dict:
            return torch.tensor(0.0), {}

        device = next(iter(attn_dict.values())).device
        dist_matrix = self._get_dist_matrix(device)

        total_gap = torch.tensor(0.0, device=device)
        total_compact = torch.tensor(0.0, device=device)
        info = {}

        for block_idx, attn_weights in attn_dict.items():
            # Compute per-head MAD — this is differentiable through attn_weights
            # attn_weights: (B, H, N, N) — N includes CLS token
            # Exclude CLS for distance computation
            attn_patch = attn_weights[:, :, 1:, 1:]  # (B, H, N_p, N_p)
            B, H, Np, _ = attn_patch.shape
            R = dist_matrix[:Np, :Np]  # (N_p, N_p)

            # MAD per head: mean over batch and query tokens
            weighted = (attn_patch * R.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (B, H, N_p)
            head_mads = weighted.mean(dim=(0, 2))  # (H,)

            # Sort indices using detached MAD (so sort order doesn't contribute gradients)
            sort_indices = head_mads.detach().argsort()
            low_indices = sort_indices[:self.num_low]
            high_indices = sort_indices[self.num_low:]

            low_mads = head_mads[low_indices]
            high_mads = head_mads[high_indices]

            mean_low = low_mads.mean()
            mean_high = high_mads.mean()
            gap = mean_high - mean_low

            # Gap loss: push groups apart
            gap_loss = torch.clamp(self.delta - gap, min=0.0)

            # Compactness loss: make each group tight
            compact_loss = low_mads.var() + high_mads.var()

            total_gap = total_gap + gap_loss
            total_compact = total_compact + compact_loss

            info[block_idx] = {
                "gap": gap.item(),
                "gap_loss": gap_loss.item(),
                "compact_loss": compact_loss.item(),
                "low_mads": low_mads.detach().cpu().tolist(),
                "high_mads": high_mads.detach().cpu().tolist(),
            }

        loss = warmup_factor * (self.lambda_gap * total_gap + self.lambda_compact * total_compact)
        info["total_gap_loss"] = total_gap.item()
        info["total_compact_loss"] = total_compact.item()
        info["total_bimodal_loss"] = loss.item()
        return loss, info
