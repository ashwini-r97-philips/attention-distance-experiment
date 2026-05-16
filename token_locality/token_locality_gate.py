"""Token-conditioned locality gate for ViT attention blocks.

Mechanism
---------
For each patch token in the input sequence, a small 1-layer branch predicts
a scalar gate value from the *normalized* token representation. The gate is
then used to inject a distance penalty into the attention logits:

    attn_logits_ij  -=  gate_i * dist(i, j)

where dist(i, j) is the spatial distance between patch i and patch j
(precomputed, shape NUM_PATCHES x NUM_PATCHES).

Design constraints:
  - CLS token (index 0) is EXEMPT from the locality penalty.
  - Gate branch is initialized near zero so the model starts as baseline.
  - Only injected into blocks 0-7 (configurable).
  - No architecture changes beyond the gate branch Linear layer.
  - Fully compatible with existing `capture_attention` hooks.

Gate branch:
    LayerNorm (shared with block's pre-norm) → Linear(embed_dim, 1) → Softplus

The gate output is non-negative (Softplus) so the distance penalty always
pulls attention toward local tokens.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.mad_metrics import build_distance_matrix


# ─── Per-patch distance matrix ────────────────────────────────────────────────

def build_patch_distance_matrix(grid: int, device: torch.device) -> torch.Tensor:
    """Build (N, N) distance matrix for a grid×grid patch layout.

    Returns normalized Euclidean distances in [0, 1].
    """
    return build_distance_matrix(grid, grid, device=device)


# ─── Gate branch ─────────────────────────────────────────────────────────────

class LocalityGateBranch(nn.Module):
    """Predicts a per-token scalar locality gate from normalized token features.

    Input:  (B, N, D) normalized token representations
    Output: (B, N, 1) non-negative gate values (Softplus activated)

    Near-zero initialization: gate_init_scale controls the weight magnitude
    so the starting penalty is negligible.
    """

    def __init__(self, embed_dim: int, gate_init_scale: float = 0.01):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1, bias=True)
        # Near-zero init so starting model behaves like baseline
        nn.init.normal_(self.linear.weight, mean=0.0, std=gate_init_scale)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_norm: (B, N, D) — pre-norm token representations
        Returns:
            gate: (B, N, 1) — non-negative gate values
        """
        return F.softplus(self.linear(x_norm))  # (B, N, 1)


# ─── Patched attention forward with locality gate ────────────────────────────

def _make_gated_attention_forward(
    gate_branch: LocalityGateBranch,
    dist_matrix: torch.Tensor,     # (N_patches, N_patches) precomputed distances
    gate_distance_scale: float,    # multiplier on penalty strength
):
    """Factory: create a replacement forward that injects the locality gate."""

    def gated_attention_forward(self, x, attn_mask=None, is_causal=False):
        """Replacement forward for timm Attention — injects locality gate bias."""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q_scaled = q * self.scale
        attn = q_scaled @ k.transpose(-2, -1)  # (B, H, N, N)

        # ── Locality gate injection ──
        # x is already the normed input from block.norm1(x) passed to attn.
        # Compute gate from it.
        gate = gate_branch(x)  # (B, N, 1)  — non-negative

        # Build penalty bias: for each query token i, add -gate_i * dist(i, j)
        # dist_matrix: (Np, Np) where Np = N - 1 (patch tokens only)
        Np = N - 1  # number of patch tokens (no CLS)
        # gate[:, 1:, 0]: (B, Np) — gate values for patch tokens only
        gate_patch = gate[:, 1:, 0]  # (B, Np)

        D = dist_matrix[:Np, :Np]    # (Np, Np) on same device

        # penalty[b, i, j] = gate_patch[b, i] * D[i, j]
        # Shape: (B, Np, Np)
        penalty = gate_patch.unsqueeze(-1) * D.unsqueeze(0)  # (B, Np, Np)
        penalty = penalty * gate_distance_scale

        # Embed into full (B, N, N) matrix with CLS rows/cols = 0 (no penalty)
        full_penalty = torch.zeros(B, N, N, dtype=attn.dtype, device=attn.device)
        full_penalty[:, 1:, 1:] = penalty

        # Subtract penalty from all heads
        # attn: (B, H, N, N)  full_penalty: (B, N, N) → broadcast over heads
        attn = attn - full_penalty.unsqueeze(1)

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = attn.softmax(dim=-1)

        # Cache for downstream capture_attention hooks
        self._cached_attn_weights = attn.detach()

        attn_dropped = self.attn_drop(attn)
        x_out = attn_dropped @ v

        x_out = x_out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        x_out = self.norm(x_out)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

    return gated_attention_forward


# ─── Block-level wrapper that injects gate ────────────────────────────────────

def _make_gated_block_forward(
    original_block,
    gate_branch: LocalityGateBranch,
    attn_module,
):
    """Create a replacement Block.forward that passes normed input to gate.

    timm Block.forward does: y = x + attn(norm1(x))
    We intercept to pass norm1(x) to the gate branch.
    """

    def gated_block_forward(self, x):
        # Pre-norm for attention
        x_normed = self.norm1(x)
        # Temporarily store normed repr for gate branch access
        attn_module._x_normed = x_normed
        # Attention (gate reads from _x_normed set above)
        x = x + self.ls1(self.drop_path1(self.attn(x_normed)))
        # Clean up
        if hasattr(attn_module, '_x_normed'):
            del attn_module._x_normed
        # FFN
        x = x + self.ls2(self.drop_path2(self.mlp(self.norm2(x))))
        return x

    return gated_block_forward


# ─── Revised approach: simpler gate that reads its own input ─────────────────
# The timm Attention module receives the *normed* token sequence directly
# (block.norm1 is applied before calling block.attn). So inside the attention
# forward, `x` IS already norm1(x). We can use it directly in the gate.
# The factory above already does this correctly — x in gated_attention_forward
# is norm1(x_block), so gate_branch(x) is correct.


# ─── Install / remove gates ───────────────────────────────────────────────────

class TokenLocalityGateModule(nn.Module):
    """Container for all per-block gate branches.

    Registered as model attribute so optimizer picks up gate parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        block_indices: list,
        embed_dim: int,
        grid: int,
        gate_init_scale: float,
        gate_distance_scale: float,
        device: torch.device,
    ):
        super().__init__()
        self.block_indices      = block_indices
        self.gate_distance_scale = gate_distance_scale

        # Precompute patch distance matrix
        dist_mat = build_patch_distance_matrix(grid, device)
        self.register_buffer("dist_matrix", dist_mat)

        # One gate branch per block
        self.gates = nn.ModuleList([
            LocalityGateBranch(embed_dim, gate_init_scale)
            for _ in block_indices
        ])

        # Patch each block's attention forward
        self._original_forwards = {}
        for i, block_idx in enumerate(block_indices):
            attn_module = model.blocks[block_idx].attn
            attn_module.fused_attn = False

            forward_fn = _make_gated_attention_forward(
                gate_branch=self.gates[i],
                dist_matrix=dist_mat,
                gate_distance_scale=gate_distance_scale,
            )
            self._original_forwards[block_idx] = attn_module.forward
            attn_module.forward = forward_fn.__get__(attn_module, type(attn_module))

    def remove_gates(self, model: nn.Module):
        """Restore original attention forwards."""
        from timm.layers.attention import Attention
        for block_idx in self.block_indices:
            attn_module = model.blocks[block_idx].attn
            if block_idx in self._original_forwards:
                attn_module.forward = self._original_forwards[block_idx]
            else:
                attn_module.forward = Attention.forward.__get__(attn_module, type(attn_module))
            attn_module.fused_attn = True

    def gate_statistics(self) -> dict:
        """Return per-layer gate weight statistics for logging."""
        stats = {}
        for i, block_idx in enumerate(self.block_indices):
            w = self.gates[i].linear.weight.detach()
            stats[f"block_{block_idx}"] = {
                "weight_mean": float(w.mean()),
                "weight_std":  float(w.std()),
                "weight_min":  float(w.min()),
                "weight_max":  float(w.max()),
                "bias":        float(self.gates[i].linear.bias.detach().item()),
            }
        return stats
