"""Locality gate modules for ADE20K segmentation.

Two conditions alongside the existing baseline:
  1. LearnedGaussianBias — LocAtViT-style: per-head learnable α_h, σ_h scalars,
     no content conditioning. Adds α_h * exp(-dist² / (2σ_h²)) to attention logits.
  2. TokenLocalityGateModuleV3 — content-conditioned, per-head signed gate (v3).
     Import directly from token_locality_gate_v3.

Both modules expose:
  - install(model)  — monkey-patches attention forwards on specified blocks
  - remove(model)   — restores original forwards
  - get_gate_params() → list[Parameter]  — for separate optimizer param group
  - gate_statistics() → dict             — for logging

Grid for ADE20K: 512px input, patch16 → 32×32 = 1024 patch tokens.
CLS token (index 0) is always exempt from the penalty.
"""

import math
import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "bimodal_head_specialisation"))

from common.mad_metrics import build_distance_matrix


def _build_dist_matrix(grid: int, device: torch.device) -> torch.Tensor:
    return build_distance_matrix(grid, grid, device=device)


# ─── LearnedGaussianBias (LocAtViT-style) ─────────────────────────────────────

class LearnedGaussianBias(nn.Module):
    """Per-head learnable Gaussian spatial bias (LocAtViT ablation).

    For each head h: bias[i,j] = α_h * exp(-dist(i,j)² / (2 * σ_h²))
    Added to attention logits (positive α → local emphasis, negative → global).

    Parameters per block: 2 * num_heads scalars (α and log_σ per head).
    Across 6 blocks × 6 heads: 72 parameters total.

    σ is parameterised as exp(log_σ) to keep it positive.
    α initialised near 0 → starts as baseline.
    log_σ initialised so σ ≈ 0.25 (mid-range spatial scale).
    """

    def __init__(self, num_heads: int, init_sigma: float = 0.25):
        super().__init__()
        self.num_heads = num_heads
        # α per head: initialised near 0 so model starts as baseline
        self.alpha = nn.Parameter(torch.zeros(num_heads))
        # log_σ per head: exp(log_σ) = init_sigma
        self.log_sigma = nn.Parameter(
            torch.full((num_heads,), math.log(init_sigma))
        )

    def compute_bias(self, dist_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dist_matrix: (Np, Np) normalised patch distances
        Returns:
            bias: (H, Np, Np) spatial bias to add to attention logits
        """
        sigma = self.log_sigma.exp()                  # (H,)
        # (H, 1, 1) broadcast over (Np, Np)
        exp_term = torch.exp(
            -dist_matrix.unsqueeze(0) ** 2 / (2 * sigma.view(-1, 1, 1) ** 2)
        )  # (H, Np, Np)
        return self.alpha.view(-1, 1, 1) * exp_term   # (H, Np, Np)


def _make_gaussian_bias_forward(
    gaussian_bias: LearnedGaussianBias,
    dist_matrix: torch.Tensor,  # (Np, Np)
):
    def gaussian_bias_forward(self, x, attn_mask=None, is_causal=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        attn = (q * self.scale) @ k.transpose(-2, -1)  # (B, H, N, N)

        # Spatial bias for patch tokens only (CLS row/col stays 0)
        Np = N - 1
        D = dist_matrix[:Np, :Np]          # (Np, Np)
        bias = gaussian_bias.compute_bias(D)  # (H, Np, Np)

        full_bias = torch.zeros(B, self.num_heads, N, N,
                                dtype=attn.dtype, device=attn.device)
        full_bias[:, :, 1:, 1:] = bias.unsqueeze(0)  # broadcast over batch

        attn = attn + full_bias

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = attn.softmax(dim=-1)
        self._cached_attn_weights = attn.detach()

        attn_dropped = self.attn_drop(attn)
        x_out = attn_dropped @ v
        x_out = x_out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        x_out = self.norm(x_out)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

    return gaussian_bias_forward


class LearnedGaussianBiasModule(nn.Module):
    """Container: installs LearnedGaussianBias on specified transformer blocks.

    Compatible with DeiTSegModel (model.blocks passthrough) and plain ViT.
    """

    def __init__(
        self,
        model: nn.Module,
        block_indices: list,
        num_heads: int,
        grid: int,
        device: torch.device,
        init_sigma: float = 0.25,
    ):
        super().__init__()
        self.block_indices = block_indices

        dist_mat = _build_dist_matrix(grid, device)
        self.register_buffer("dist_matrix", dist_mat)

        self.biases = nn.ModuleList([
            LearnedGaussianBias(num_heads, init_sigma=init_sigma)
            for _ in block_indices
        ])

        self._original_forwards = {}
        for i, block_idx in enumerate(block_indices):
            attn_module = model.blocks[block_idx].attn
            attn_module.fused_attn = False
            forward_fn = _make_gaussian_bias_forward(
                gaussian_bias=self.biases[i],
                dist_matrix=dist_mat,
            )
            self._original_forwards[block_idx] = attn_module.forward
            attn_module.forward = forward_fn.__get__(attn_module, type(attn_module))

    def remove(self, model: nn.Module):
        for block_idx in self.block_indices:
            attn_module = model.blocks[block_idx].attn
            if block_idx in self._original_forwards:
                attn_module.forward = self._original_forwards[block_idx]
            attn_module.fused_attn = True

    def get_gate_params(self) -> list:
        return list(self.parameters())

    def gate_statistics(self) -> dict:
        stats = {}
        for i, block_idx in enumerate(self.block_indices):
            b = self.biases[i]
            stats[f"block_{block_idx}"] = {
                "alpha_mean": float(b.alpha.detach().mean()),
                "alpha_min": float(b.alpha.detach().min()),
                "alpha_max": float(b.alpha.detach().max()),
                "sigma_mean": float(b.log_sigma.exp().detach().mean()),
                "sigma_min": float(b.log_sigma.exp().detach().min()),
                "sigma_max": float(b.log_sigma.exp().detach().max()),
            }
        return stats
