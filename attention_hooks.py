"""Hooks to extract attention weights from timm ViT models.

The main challenge: timm's Attention uses F.scaled_dot_product_attention (fused)
by default, which does not expose attention weights. We handle this by replacing
the forward method on targeted blocks.
"""

import torch
import torch.nn.functional as F
from contextlib import contextmanager

from timm.layers.attention import Attention


def _attention_forward_with_weights(self, x, attn_mask=None, is_causal=False):
    """Replacement forward that computes and caches attention weights."""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)  # (B, H, N, N)

    if attn_mask is not None:
        attn = attn + attn_mask

    attn = attn.softmax(dim=-1)

    # Cache the attention weights
    self._cached_attn_weights = attn.detach()

    attn_dropped = self.attn_drop(attn)
    x = attn_dropped @ v

    x = x.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
    x = self.norm(x)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def _attention_forward_with_weights_differentiable(self, x, attn_mask=None, is_causal=False):
    """Replacement forward that caches attention weights WITH gradients for training."""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)  # (B, H, N, N)

    if attn_mask is not None:
        attn = attn + attn_mask

    attn = attn.softmax(dim=-1)

    # Cache WITH gradients so the bimodal loss can backprop through MAD
    self._cached_attn_weights = attn

    attn_dropped = self.attn_drop(attn)
    x = attn_dropped @ v

    x = x.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
    x = self.norm(x)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def patch_attention_forward(model, block_indices, differentiable=False):
    """Replace attention forward on specified blocks to expose attention weights.

    Args:
        model: VisionTransformer model.
        block_indices: list of block indices to patch.
        differentiable: if True, cache attn weights with gradients (for training).
    """
    forward_fn = (
        _attention_forward_with_weights_differentiable if differentiable
        else _attention_forward_with_weights
    )
    for idx in block_indices:
        attn_module = model.blocks[idx].attn
        attn_module.forward = forward_fn.__get__(attn_module, type(attn_module))
        attn_module.fused_attn = False


def unpatch_attention_forward(model, block_indices):
    """Restore original attention forward on specified blocks."""
    for idx in block_indices:
        attn_module = model.blocks[idx].attn
        # Restore original forward from the class
        if hasattr(attn_module, '_cached_attn_weights'):
            del attn_module._cached_attn_weights
        attn_module.forward = Attention.forward.__get__(attn_module, type(attn_module))
        attn_module.fused_attn = True


def get_cached_attn_weights(model, block_indices):
    """Retrieve cached attention weights from patched blocks.

    Returns:
        dict: {block_idx: attn_weights tensor (B, H, N, N)}
    """
    result = {}
    for idx in block_indices:
        attn_module = model.blocks[idx].attn
        if hasattr(attn_module, '_cached_attn_weights'):
            result[idx] = attn_module._cached_attn_weights
    return result


def clear_cached_attn_weights(model, block_indices):
    """Clear cached attention weights to free memory."""
    for idx in block_indices:
        attn_module = model.blocks[idx].attn
        if hasattr(attn_module, '_cached_attn_weights'):
            del attn_module._cached_attn_weights


@contextmanager
def capture_attention(model, block_indices=None):
    """Context manager for attention capture during eval.

    Usage:
        with capture_attention(model, [0, 1, 2]) as get_attn:
            output = model(images)
            attn_dict = get_attn()
    """
    if block_indices is None:
        block_indices = list(range(len(model.blocks)))

    patch_attention_forward(model, block_indices, differentiable=False)
    try:
        def get_attn():
            return get_cached_attn_weights(model, block_indices)
        yield get_attn
    finally:
        unpatch_attention_forward(model, block_indices)
