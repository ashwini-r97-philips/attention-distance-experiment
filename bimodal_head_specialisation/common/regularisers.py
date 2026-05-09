"""Attention-distance regularisers: spread loss and bimodal mixture loss."""

import torch
import torch.nn as nn
import math

from .mad_metrics import build_distance_matrix


class SpreadLoss(nn.Module):
    """Encourage heads within each layer to have diverse MAD values.

    L_spread = - sum_l Var_h(d_lh)
    """

    def __init__(self, grid_h, grid_w, device="cpu"):
        super().__init__()
        self.register_buffer("dist_matrix", build_distance_matrix(grid_h, grid_w, device))

    def _head_mads(self, attn_weights):
        """Compute per-head MAD from (B, H, N, N) attention, excluding CLS."""
        a = attn_weights[:, :, 1:, 1:]
        B, H, Np, _ = a.shape
        R = self.dist_matrix[:Np, :Np]
        weighted = (a * R.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        return weighted.mean(dim=(0, 2))  # (H,)

    def forward(self, attn_dict, warmup_factor=1.0):
        """
        Args:
            attn_dict: {block_idx: attn_weights (B, H, N, N)} with gradients.
            warmup_factor: float in [0, 1].
        Returns:
            loss: scalar
            info: dict for logging
        """
        if not attn_dict:
            device = self.dist_matrix.device
            return torch.tensor(0.0, device=device), {}

        device = next(iter(attn_dict.values())).device
        total_neg_var = torch.tensor(0.0, device=device)
        info = {}

        for block_idx, attn_weights in attn_dict.items():
            mads = self._head_mads(attn_weights)
            var = mads.var()
            total_neg_var = total_neg_var - var
            info[block_idx] = {
                "mad_variance": var.item(),
                "head_mads": mads.detach().cpu().tolist(),
            }

        loss = warmup_factor * total_neg_var
        info["total_spread_loss"] = loss.item()
        return loss, info


class BimodalMixtureLoss(nn.Module):
    """Bimodal Gaussian mixture prior on per-head MAD.

    L_bi = -sum_{l,h} log( pi * N(d_lh; m_loc, sigma_loc)
                         + (1-pi) * N(d_lh; m_glob, sigma_glob) + eps )
    """

    def __init__(self, grid_h, grid_w, m_loc=0.20, m_glob=0.65,
                 sigma_loc=0.08, sigma_glob=0.12, pi_mix=0.5, device="cpu"):
        super().__init__()
        self.register_buffer("dist_matrix", build_distance_matrix(grid_h, grid_w, device))
        self.m_loc = m_loc
        self.m_glob = m_glob
        self.sigma_loc = sigma_loc
        self.sigma_glob = sigma_glob
        self.pi_mix = pi_mix

    @staticmethod
    def _log_normal(x, mu, sigma):
        """Log pdf of univariate normal."""
        return -0.5 * math.log(2 * math.pi) - math.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2

    def _head_mads(self, attn_weights):
        a = attn_weights[:, :, 1:, 1:]
        B, H, Np, _ = a.shape
        R = self.dist_matrix[:Np, :Np]
        weighted = (a * R.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        return weighted.mean(dim=(0, 2))

    def forward(self, attn_dict, warmup_factor=1.0):
        if not attn_dict:
            device = self.dist_matrix.device
            return torch.tensor(0.0, device=device), {}

        device = next(iter(attn_dict.values())).device
        total_nll = torch.tensor(0.0, device=device)
        info = {}
        eps = 1e-8

        for block_idx, attn_weights in attn_dict.items():
            mads = self._head_mads(attn_weights)  # (H,)

            log_p_loc = self._log_normal(mads, self.m_loc, self.sigma_loc)
            log_p_glob = self._log_normal(mads, self.m_glob, self.sigma_glob)

            # Log-sum-exp for numerical stability
            log_pi = math.log(self.pi_mix + eps)
            log_one_minus_pi = math.log(1.0 - self.pi_mix + eps)

            log_mix = torch.logaddexp(
                log_pi + log_p_loc,
                log_one_minus_pi + log_p_glob,
            )
            nll = -log_mix.sum()
            total_nll = total_nll + nll

            info[block_idx] = {
                "nll": nll.item(),
                "head_mads": mads.detach().cpu().tolist(),
            }

        loss = warmup_factor * total_nll
        info["total_bimodal_loss"] = loss.item()
        return loss, info


def build_regulariser(cfg):
    """Factory: build the right regulariser from config.

    Returns:
        regulariser module (or None if reg_type == 'none')
    """
    if cfg.reg_type == "none":
        return None

    device = cfg.device
    grid_h, grid_w = cfg.grid_h, cfg.grid_w

    if cfg.reg_type == "spread":
        return SpreadLoss(grid_h, grid_w, device=device)

    if cfg.reg_type == "bimodal":
        return BimodalMixtureLoss(
            grid_h, grid_w,
            m_loc=cfg.m_loc, m_glob=cfg.m_glob,
            sigma_loc=cfg.sigma_loc, sigma_glob=cfg.sigma_glob,
            pi_mix=cfg.pi_mix, device=device,
        )

    raise ValueError(f"Unknown reg_type: {cfg.reg_type}")
