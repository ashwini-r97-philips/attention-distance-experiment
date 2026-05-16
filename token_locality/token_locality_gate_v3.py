"""Token-conditioned locality gate v3 — per-head, signed, bidirectional.

Key changes from v2
-------------------
1. **Per-head gate** — Linear(D → H) instead of Linear(D → 1).
   Each head gets its own scalar per token, so heads can independently
   specialise as local or global within the same block.

2. **Signed activation (scale * tanh)** — replaces Softplus.
   - Positive gate  → subtracts from logits for distant patches (local)
   - Negative gate  → adds to logits for distant patches (global)
   - |tanh| ≤ 1, scaled by `gate_scale` (default 2.0) for controlled range
   - Near-zero at init because bias=-0.0 and weights are tiny

3. **Bias initialised to 0.0** — tanh(0) = 0, so starting penalty is zero
   for all tokens across all heads. The model starts as unmodified baseline.
   (v2 needed bias=-5 to fight Softplus(0)=0.69; tanh(0)=0 doesn't need that.)

The penalty applied to attention logits before softmax:
    penalty[b, h, i, j] = gate[b, i, h] * dist(i, j) * gate_distance_scale

where gate[b, i, h] ∈ (-gate_scale, +gate_scale) is the per-token per-head value.

CLS token (index 0) is exempt — its rows/cols are always zero penalty.

─────────────────────────────────────────────────────────────────────────────
VectorKernelGate  [NOT WIRED INTO TRAINING — ready for future integration]
─────────────────────────────────────────────────────────────────────────────
A more expressive variant where the gate learns a full distance profile
instead of scaling a fixed scalar distance.

Each token outputs a K-dim vector. The distance between patch i and patch j
is encoded into K features via a fixed RBF basis. The penalty is the dot
product between the token's gate vector and the distance feature vector:

    penalty[b, h, i, j] = Σ_k  gate_vec[b, i, h, k] * rbf_k(dist(i,j))

This lets a token say e.g. "boost mid-range attention, suppress very far"
— arbitrary learned distance profiles, not just linear scaling.

To integrate: replace LocalityGateBranchV3 with VectorKernelGateBranch and
swap _make_gated_attention_forward_v3 for _make_vector_kernel_forward.
The module container (TokenLocalityGateModuleV3) accepts a branch_factory
argument so the swap is one line.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.mad_metrics import build_distance_matrix


# ─── Distance matrix ─────────────────────────────────────────────────────────

def build_patch_distance_matrix(grid: int, device: torch.device) -> torch.Tensor:
    return build_distance_matrix(grid, grid, device=device)


# ─── RBF distance feature encoder (shared utility) ───────────────────────────

class RBFDistanceEncoder(nn.Module):
    """Encode scalar distances into a K-dim RBF feature vector.

    Fixed (non-learnable) encoder — the gate branch learns what to do
    with the features, not the features themselves.

    centres are evenly spaced in [0, 1]; bandwidth = spacing between centres.

    Shape: (Np, Np) → (Np, Np, K)
    """

    def __init__(self, num_basis: int = 16):
        super().__init__()
        self.num_basis = num_basis
        centres = torch.linspace(0.0, 1.0, num_basis)  # (K,)
        bandwidth = 1.0 / (num_basis - 1)
        self.register_buffer("centres", centres)
        self.register_buffer("bandwidth", torch.tensor(bandwidth))

    def forward(self, dist_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dist_matrix: (Np, Np) normalised distances in [0, 1]
        Returns:
            (Np, Np, K) RBF features, values in (0, 1]
        """
        # (Np, Np, 1) - (1, 1, K) → (Np, Np, K)
        diff = dist_matrix.unsqueeze(-1) - self.centres.view(1, 1, -1)
        return torch.exp(-0.5 * (diff / self.bandwidth) ** 2)


# ─── Gate branch v3: per-head, signed ────────────────────────────────────────

class LocalityGateBranchV3(nn.Module):
    """Per-token, per-head scalar gate with signed (tanh) output.

    Input:  (B, N, D) normalised token representations
    Output: (B, N, H) signed gate values in (-gate_scale, +gate_scale)

    Positive → local bias for that head/token pair
    Negative → global bias for that head/token pair

    Init: weights ~ N(0, 0.02), bias = 0.0  → gate ≈ 0 at start (baseline)
    """

    def __init__(self, embed_dim: int, num_heads: int, gate_scale: float = 2.0,
                 weight_std: float = 0.02):
        super().__init__()
        self.num_heads = num_heads
        self.gate_scale = gate_scale
        self.linear = nn.Linear(embed_dim, num_heads, bias=True)
        nn.init.normal_(self.linear.weight, mean=0.0, std=weight_std)
        nn.init.zeros_(self.linear.bias)
        # Tracked for logging
        self._last_output_mean: float = 0.0
        self._last_output_std: float = 0.0

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_norm: (B, N, D)
        Returns:
            gate: (B, N, H) values in (-gate_scale, +gate_scale)
        """
        out = self.gate_scale * torch.tanh(self.linear(x_norm))  # (B, N, H)
        if self.training:
            with torch.no_grad():
                patch_out = out[:, 1:, :]  # exclude CLS
                self._last_output_mean = patch_out.mean().item()
                self._last_output_std = patch_out.std().item()
        return out


# ─── Patched attention forward v3 ────────────────────────────────────────────

def _make_gated_attention_forward_v3(
    gate_branch: LocalityGateBranchV3,
    dist_matrix: torch.Tensor,  # (Np, Np)
    gate_distance_scale: float,
):
    """Factory: per-head signed locality gate injected before softmax."""

    def gated_attention_forward_v3(self, x, attn_mask=None, is_causal=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        attn = (q * self.scale) @ k.transpose(-2, -1)  # (B, H, N, N)

        # ── Per-head signed gate ──────────────────────────────────────────────
        gate = gate_branch(x)           # (B, N, H)
        Np = N - 1                      # patch tokens only
        gate_patch = gate[:, 1:, :]     # (B, Np, H)
        D = dist_matrix[:Np, :Np]       # (Np, Np)

        # penalty[b, i, h, j] = gate_patch[b, i, h] * D[i, j] * scale
        # gate_patch: (B, Np, H)  →  (B, Np, H, 1)
        # D:          (Np, Np)    →  (1, Np, 1, Np)
        penalty = gate_patch.unsqueeze(-1) * D.unsqueeze(0).unsqueeze(2)
        # (B, Np, H, Np) → (B, H, Np, Np)
        penalty = penalty.permute(0, 2, 1, 3) * gate_distance_scale

        # Embed into full (B, H, N, N) with CLS rows/cols = 0
        full_penalty = torch.zeros(B, self.num_heads, N, N,
                                   dtype=attn.dtype, device=attn.device)
        full_penalty[:, :, 1:, 1:] = penalty

        attn = attn - full_penalty      # (B, H, N, N)

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

    return gated_attention_forward_v3


# ─── Module container v3 ─────────────────────────────────────────────────────

class TokenLocalityGateModuleV3(nn.Module):
    """Container for v3 per-block, per-head, signed gate branches.

    reg_type: 'token_locality_v3'

    Constructor accepts a branch_factory callable so the attention forward
    factory can be swapped out without touching this class — used by the
    VectorKernelGateModule below.
    """

    def __init__(
        self,
        model: nn.Module,
        block_indices: list,
        embed_dim: int,
        num_heads: int,
        grid: int,
        gate_distance_scale: float,
        gate_scale: float,
        device: torch.device,
        weight_std: float = 0.02,
    ):
        super().__init__()
        self.block_indices = block_indices
        self.gate_distance_scale = gate_distance_scale

        dist_mat = build_patch_distance_matrix(grid, device)
        self.register_buffer("dist_matrix", dist_mat)

        self.gates = nn.ModuleList([
            LocalityGateBranchV3(embed_dim, num_heads, gate_scale=gate_scale,
                                  weight_std=weight_std)
            for _ in block_indices
        ])

        self._original_forwards = {}
        for i, block_idx in enumerate(block_indices):
            attn_module = model.blocks[block_idx].attn
            attn_module.fused_attn = False
            forward_fn = _make_gated_attention_forward_v3(
                gate_branch=self.gates[i],
                dist_matrix=dist_mat,
                gate_distance_scale=gate_distance_scale,
            )
            self._original_forwards[block_idx] = attn_module.forward
            attn_module.forward = forward_fn.__get__(attn_module, type(attn_module))

    def remove_gates(self, model: nn.Module):
        for block_idx in self.block_indices:
            attn_module = model.blocks[block_idx].attn
            if block_idx in self._original_forwards:
                attn_module.forward = self._original_forwards[block_idx]
            attn_module.fused_attn = True

    def get_gate_params(self):
        return list(self.parameters())

    def gate_statistics(self) -> dict:
        stats = {}
        for i, block_idx in enumerate(self.block_indices):
            gate = self.gates[i]
            w = gate.linear.weight.detach()
            b = gate.linear.bias.detach()
            stats[f"block_{block_idx}"] = {
                "weight_mean": float(w.mean()),
                "weight_std": float(w.std()),
                "weight_min": float(w.min()),
                "weight_max": float(w.max()),
                "bias_mean": float(b.mean()),
                "bias_std": float(b.std()),
                "gate_output_mean": gate._last_output_mean,
                "gate_output_std": gate._last_output_std,
            }
        return stats


# ═════════════════════════════════════════════════════════════════════════════
# VectorKernelGate  —  NOT wired into training, ready for future integration
# ═════════════════════════════════════════════════════════════════════════════

class VectorKernelGateBranch(nn.Module):
    """Per-token, per-head vector gate with learned distance profile.

    Instead of scaling a fixed scalar distance, the gate outputs a K-dim
    vector per (token, head). The penalty is the dot product with K RBF
    features of the actual patch distance:

        penalty[b, h, i, j] = Σ_k gate_vec[b, i, h, k] * rbf_k(dist(i,j))

    This allows each token to learn an arbitrary distance tuning curve
    (e.g. "boost mid-range, suppress far") rather than a simple linear scale.

    Architecture: 2-layer MLP
        D → hidden_dim (GELU) → H * K (split per head, per basis)

    Signed output: tanh * gate_scale on the final projection, same as v3.

    To integrate into training:
      1. Replace LocalityGateBranchV3 with VectorKernelGateBranch in the
         gate module constructor.
      2. Swap _make_gated_attention_forward_v3 for _make_vector_kernel_forward
         (defined below).
      3. Pre-compute rbf_features once in VectorKernelGateModule.__init__ and
         register as a buffer.
      4. Add reg_type: 'token_locality_v3_vec' to configs and train_cub.py.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_basis: int = 16,
        hidden_dim: int = 64,
        gate_scale: float = 2.0,
        weight_std: float = 0.02,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_basis = num_basis
        self.gate_scale = gate_scale

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads * num_basis, bias=True),
        )
        # Init: near-zero weights so output ≈ 0 at start
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=weight_std)
                nn.init.zeros_(layer.bias)

        self._last_output_mean: float = 0.0
        self._last_output_std: float = 0.0

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_norm: (B, N, D)
        Returns:
            gate_vec: (B, N, H, K) signed gate vectors
        """
        B, N, D = x_norm.shape
        out = self.mlp(x_norm)                           # (B, N, H*K)
        out = self.gate_scale * torch.tanh(out)
        out = out.view(B, N, self.num_heads, self.num_basis)  # (B, N, H, K)
        if self.training:
            with torch.no_grad():
                patch_out = out[:, 1:, :, :]
                self._last_output_mean = patch_out.mean().item()
                self._last_output_std = patch_out.std().item()
        return out


def _make_vector_kernel_forward(
    gate_branch: VectorKernelGateBranch,
    dist_matrix: torch.Tensor,    # (Np, Np)
    rbf_features: torch.Tensor,   # (Np, Np, K) — precomputed, registered as buffer
    gate_distance_scale: float,
):
    """Factory: attention forward with vector-kernel locality gate.

    penalty[b, h, i, j] = (Σ_k gate_vec[b,i,h,k] * rbf[i,j,k]) * scale
    """

    def vector_kernel_attention_forward(self, x, attn_mask=None, is_causal=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        attn = (q * self.scale) @ k.transpose(-2, -1)  # (B, H, N, N)

        gate_vec = gate_branch(x)       # (B, N, H, K)
        Np = N - 1
        gate_patch = gate_vec[:, 1:, :, :]  # (B, Np, H, K)
        rbf = rbf_features[:Np, :Np, :]     # (Np, Np, K)

        # penalty[b, i, h, j] = Σ_k gate_patch[b,i,h,k] * rbf[i,j,k]
        # gate_patch: (B, Np, H, K)  rbf: (Np, Np, K)
        # einsum: b i h k, i j k -> b i h j  → (B, Np, H, Np)
        penalty = torch.einsum("bihk,ijk->bihj", gate_patch, rbf)
        # (B, Np, H, Np) → (B, H, Np, Np)
        penalty = penalty.permute(0, 2, 1, 3) * gate_distance_scale

        full_penalty = torch.zeros(B, self.num_heads, N, N,
                                   dtype=attn.dtype, device=attn.device)
        full_penalty[:, :, 1:, 1:] = penalty

        attn = attn - full_penalty
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

    return vector_kernel_attention_forward


class VectorKernelGateModule(nn.Module):
    """Container for vector-kernel gate branches (future integration).

    Usage (when ready to plug in):
        gate_module = VectorKernelGateModule(
            model=model,
            block_indices=cfg.regularized_blocks,
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            grid=cfg.grid_h,
            gate_distance_scale=cfg.gate_distance_scale,
            gate_scale=cfg.gate_scale,
            num_basis=16,
            hidden_dim=64,
            device=device,
        )
    Then add 'token_locality_v3_vec' as a reg_type in train_cub.py.
    """

    def __init__(
        self,
        model: nn.Module,
        block_indices: list,
        embed_dim: int,
        num_heads: int,
        grid: int,
        gate_distance_scale: float,
        gate_scale: float,
        device: torch.device,
        num_basis: int = 16,
        hidden_dim: int = 64,
        weight_std: float = 0.02,
    ):
        super().__init__()
        self.block_indices = block_indices

        dist_mat = build_patch_distance_matrix(grid, device)
        self.register_buffer("dist_matrix", dist_mat)

        rbf_encoder = RBFDistanceEncoder(num_basis=num_basis)
        rbf_encoder = rbf_encoder.to(device)
        rbf_feats = rbf_encoder(dist_mat)   # (Np, Np, K) — precomputed once
        self.register_buffer("rbf_features", rbf_feats)

        self.gates = nn.ModuleList([
            VectorKernelGateBranch(embed_dim, num_heads, num_basis=num_basis,
                                   hidden_dim=hidden_dim, gate_scale=gate_scale,
                                   weight_std=weight_std)
            for _ in block_indices
        ])

        self._original_forwards = {}
        for i, block_idx in enumerate(block_indices):
            attn_module = model.blocks[block_idx].attn
            attn_module.fused_attn = False
            forward_fn = _make_vector_kernel_forward(
                gate_branch=self.gates[i],
                dist_matrix=dist_mat,
                rbf_features=rbf_feats,
                gate_distance_scale=gate_distance_scale,
            )
            self._original_forwards[block_idx] = attn_module.forward
            attn_module.forward = forward_fn.__get__(attn_module, type(attn_module))

    def remove_gates(self, model: nn.Module):
        for block_idx in self.block_indices:
            attn_module = model.blocks[block_idx].attn
            if block_idx in self._original_forwards:
                attn_module.forward = self._original_forwards[block_idx]
            attn_module.fused_attn = True

    def get_gate_params(self):
        return list(self.parameters())

    def gate_statistics(self) -> dict:
        stats = {}
        for i, block_idx in enumerate(self.block_indices):
            gate = self.gates[i]
            stats[f"block_{block_idx}"] = {
                "gate_output_mean": gate._last_output_mean,
                "gate_output_std": gate._last_output_std,
            }
        return stats
