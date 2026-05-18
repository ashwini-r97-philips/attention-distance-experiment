"""Attention weight capture hooks for eval-time MAD logging.

Copied and made self-contained from bimodal_head_specialisation/common/attention_hooks.py.
Only the eval-time (non-differentiable) capture path is included here, since
the token locality gates handle their own attention interception during training.
"""

from contextlib import contextmanager
import torch
from timm.layers.attention import Attention


def _attention_forward_with_weights(self, x, attn_mask=None, is_causal=False):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = attn.softmax(dim=-1)
    self._cached_attn_weights = attn.detach()

    attn_dropped = self.attn_drop(attn)
    x = attn_dropped @ v
    x = x.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
    x = self.norm(x)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def _patch(model, block_indices):
    for idx in block_indices:
        attn = model.blocks[idx].attn
        attn.fused_attn = False
        attn.forward = _attention_forward_with_weights.__get__(attn, type(attn))


def _unpatch(model, block_indices):
    for idx in block_indices:
        attn = model.blocks[idx].attn
        if hasattr(attn, "_cached_attn_weights"):
            del attn._cached_attn_weights
        attn.forward = Attention.forward.__get__(attn, type(attn))
        attn.fused_attn = True


def _get_cached(model, block_indices):
    return {
        idx: model.blocks[idx].attn._cached_attn_weights
        for idx in block_indices
        if hasattr(model.blocks[idx].attn, "_cached_attn_weights")
    }


@contextmanager
def capture_attention(model, block_indices=None):
    """Context manager: capture attention weights from specified blocks (no grad)."""
    if block_indices is None:
        block_indices = list(range(len(model.blocks)))
    _patch(model, block_indices)
    try:
        yield lambda: _get_cached(model, block_indices)
    finally:
        _unpatch(model, block_indices)
