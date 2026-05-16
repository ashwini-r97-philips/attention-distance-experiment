"""Token-conditioned locality gate v2 — fixed learning dynamics.

Key changes from v1:
  1. Bias initialized to -5.0 so Softplus(-5) ≈ 0.007 → starts as near-baseline
  2. Separate (higher) learning rate for gate parameters
  3. Logs gate OUTPUT statistics, not just weight statistics
  4. Added FixedLocalityPrior for ablation (no learnable parameters)

The penalty is:  gate(token_i) * distance(i, j) * gate_distance_scale
where gate(token_i) = Softplus(Linear(x_i) + bias)

With bias=-5.0 init, starting penalty ≈ 0.007 × dist × scale ≈ negligible.
The gate must LEARN to increase from zero, rather than being stuck near 0.69.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.mad_metrics import build_distance_matrix


# ─── Distance matrix ─────────────────────────────────────────────────────────

def build_patch_distance_matrix(grid: int, device: torch.device) -> torch.Tensor:
    """Build (N, N) normalized distance matrix for grid×grid patches."""
    return build_distance_matrix(grid, grid, device=device)


# ─── Fixed Locality Prior (no learnable parameters) ──────────────────────────

class FixedLocalityPrior(nn.Module):
    """Applies a fixed distance penalty to attention logits.
    
    penalty_ij = fixed_strength * distance(i, j)
    
    No learnable parameters. Used as ablation to test whether
    the adaptive gate adds value beyond a constant penalty.
    """

    def __init__(
        self,
        model: nn.Module,
        block_indices: list,
        grid: int,
        fixed_strength: float,
        gate_distance_scale: float,
        device: torch.device,
    ):
        super().__init__()
        self.block_indices = block_indices
        self.gate_distance_scale = gate_distance_scale
        self.fixed_strength = fixed_strength

        dist_mat = build_patch_distance_matrix(grid, device)
        self.register_buffer("dist_matrix", dist_mat)

        # Install fixed penalty on each block
        self._original_forwards = {}
        for block_idx in block_indices:
            attn_module = model.blocks[block_idx].attn
            attn_module.fused_attn = False

            forward_fn = _make_fixed_penalty_forward(
                dist_matrix=dist_mat,
                fixed_strength=fixed_strength,
                gate_distance_scale=gate_distance_scale,
            )
            self._original_forwards[block_idx] = attn_module.forward
            attn_module.forward = forward_fn.__get__(attn_module, type(attn_module))

    def remove_gates(self, model: nn.Module):
        """Restore original attention forwards."""
        for block_idx in self.block_indices:
            attn_module = model.blocks[block_idx].attn
            if block_idx in self._original_forwards:
                attn_module.forward = self._original_forwards[block_idx]
            attn_module.fused_attn = True

    def gate_statistics(self):
        """Return empty stats (no learnable params)."""
        return {f"block_{b}": {
            "weight_mean": 0.0, "weight_std": 0.0,
            "weight_min": 0.0, "weight_max": 0.0, "bias": 0.0,
            "gate_output_mean": self.fixed_strength,
            "gate_output_std": 0.0,
        } for b in self.block_indices}


def _make_fixed_penalty_forward(dist_matrix, fixed_strength, gate_distance_scale):
    """Factory: attention forward with fixed (non-learnable) distance penalty."""

    # Precompute the full penalty matrix (constant across all inputs)
    Np = dist_matrix.shape[0]
    penalty_patch = fixed_strength * gate_distance_scale * dist_matrix[:Np, :Np]

    def fixed_penalty_attention_forward(self, x, attn_mask=None, is_causal=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        attn = (q * self.scale) @ k.transpose(-2, -1)  # (B, H, N, N)

        # Apply fixed penalty to patch-to-patch attention only
        Np_cur = N - 1
        full_penalty = torch.zeros(N, N, dtype=attn.dtype, device=attn.device)
        full_penalty[1:, 1:] = penalty_patch[:Np_cur, :Np_cur]
        attn = attn - full_penalty  # broadcast (N,N) over (B,H,N,N)

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

    return fixed_penalty_attention_forward


# ─── Learnable Gate v2 ───────────────────────────────────────────────────────

class LocalityGateBranchV2(nn.Module):
    """Per-token locality gate with proper initialization.

    Key difference from v1:
      - bias initialized to -5.0 → Softplus(-5) ≈ 0.007 (near-zero start)
      - weight initialized with larger std for expressiveness
      - Logs output statistics for monitoring

    This means the model starts as baseline and must LEARN to apply locality.
    """

    def __init__(self, embed_dim: int, init_bias: float = -5.0, weight_std: float = 0.02):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1, bias=True)
        nn.init.normal_(self.linear.weight, mean=0.0, std=weight_std)
        nn.init.constant_(self.linear.bias, init_bias)
        self._last_output_mean = 0.0
        self._last_output_std = 0.0

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_norm: (B, N, D)
        Returns:
            gate: (B, N, 1) non-negative
        """
        out = F.softplus(self.linear(x_norm))  # (B, N, 1)
        # Track output stats (detached, cheap)
        if self.training:
            with torch.no_grad():
                patch_out = out[:, 1:, 0]  # exclude CLS
                self._last_output_mean = patch_out.mean().item()
                self._last_output_std = patch_out.std().item()
        return out


# ─── Patched attention forward with v2 gate ──────────────────────────────────

def _make_gated_attention_forward_v2(
    gate_branch: LocalityGateBranchV2,
    dist_matrix: torch.Tensor,
    gate_distance_scale: float,
):
    """Factory: replacement forward with adaptive locality gate."""

    def gated_attention_forward_v2(self, x, attn_mask=None, is_causal=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        attn = (q * self.scale) @ k.transpose(-2, -1)  # (B, H, N, N)

        # Adaptive locality gate
        gate = gate_branch(x)  # (B, N, 1)
        Np = N - 1
        gate_patch = gate[:, 1:, 0]  # (B, Np)
        D = dist_matrix[:Np, :Np]

        # penalty[b, i, j] = gate_patch[b, i] * D[i, j] * scale
        penalty = gate_patch.unsqueeze(-1) * D.unsqueeze(0)
        penalty = penalty * gate_distance_scale

        full_penalty = torch.zeros(B, N, N, dtype=attn.dtype, device=attn.device)
        full_penalty[:, 1:, 1:] = penalty
        attn = attn - full_penalty.unsqueeze(1)

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

    return gated_attention_forward_v2


# ─── Module container for v2 gate ────────────────────────────────────────────

class TokenLocalityGateModuleV2(nn.Module):
    """Container for v2 per-block gate branches.

    Changes from v1:
      - Uses LocalityGateBranchV2 (bias=-5, larger weight std)
      - Reports gate output statistics
      - Supports separate LR via get_gate_params()
    """

    def __init__(
        self,
        model: nn.Module,
        block_indices: list,
        embed_dim: int,
        grid: int,
        gate_distance_scale: float,
        device: torch.device,
        init_bias: float = -5.0,
        weight_std: float = 0.02,
    ):
        super().__init__()
        self.block_indices = block_indices
        self.gate_distance_scale = gate_distance_scale

        dist_mat = build_patch_distance_matrix(grid, device)
        self.register_buffer("dist_matrix", dist_mat)

        self.gates = nn.ModuleList([
            LocalityGateBranchV2(embed_dim, init_bias=init_bias, weight_std=weight_std)
            for _ in block_indices
        ])

        self._original_forwards = {}
        for i, block_idx in enumerate(block_indices):
            attn_module = model.blocks[block_idx].attn
            attn_module.fused_attn = False

            forward_fn = _make_gated_attention_forward_v2(
                gate_branch=self.gates[i],
                dist_matrix=dist_mat,
                gate_distance_scale=gate_distance_scale,
            )
            self._original_forwards[block_idx] = attn_module.forward
            attn_module.forward = forward_fn.__get__(attn_module, type(attn_module))

    def remove_gates(self, model: nn.Module):
        """Restore original attention forwards."""
        for block_idx in self.block_indices:
            attn_module = model.blocks[block_idx].attn
            if block_idx in self._original_forwards:
                attn_module.forward = self._original_forwards[block_idx]
            attn_module.fused_attn = True

    def get_gate_params(self):
        """Return gate parameters for separate optimizer group."""
        return list(self.parameters())

    def gate_statistics(self) -> dict:
        """Return per-layer gate statistics including OUTPUT magnitudes."""
        stats = {}
        for i, block_idx in enumerate(self.block_indices):
            gate = self.gates[i]
            w = gate.linear.weight.detach()
            stats[f"block_{block_idx}"] = {
                "weight_mean": float(w.mean()),
                "weight_std": float(w.std()),
                "weight_min": float(w.min()),
                "weight_max": float(w.max()),
                "bias": float(gate.linear.bias.detach().item()),
                "gate_output_mean": gate._last_output_mean,
                "gate_output_std": gate._last_output_std,
            }
        return stats
