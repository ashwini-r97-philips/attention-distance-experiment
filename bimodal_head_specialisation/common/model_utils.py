"""Model utilities: load DeiT-S, prepare for training, head masking."""

import os

import torch
import timm

from . import config as cfg
from .attention_hooks import patch_attention_forward, get_cached_attn_weights, clear_cached_attn_weights

# Direct URL for DeiT-S weights (bypasses HuggingFace SSL issues)
DEIT_S_URL = "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth"
DEIT_S_CACHE = os.path.expanduser("~/.cache/torch/hub/checkpoints/deit_small_patch16_224-cd65a155.pth")


def load_deit_small(pretrained=True, device=None):
    """Load DeiT-S, using locally cached Facebook weights if HF download fails."""
    device = device or cfg.DEVICE
    # Create model without pretrained weights first
    model = timm.create_model(cfg.MODEL_NAME, pretrained=False, num_classes=cfg.NUM_CLASSES)

    if pretrained:
        if os.path.exists(DEIT_S_CACHE):
            ckpt = torch.load(DEIT_S_CACHE, map_location="cpu", weights_only=True)
            state_dict = ckpt["model"] if "model" in ckpt else ckpt
            # Filter out head weights if num_classes differs, and dist_token if present
            model_keys = set(model.state_dict().keys())
            filtered = {k: v for k, v in state_dict.items() if k in model_keys and v.shape == model.state_dict()[k].shape}
            missing = model_keys - set(filtered.keys())
            if missing:
                print(f"  Loading pretrained: {len(filtered)} matched, {len(missing)} missing (will use init): {missing}")
            model.load_state_dict(filtered, strict=False)
        else:
            # Fallback: try timm's download (may fail with SSL)
            model = timm.create_model(cfg.MODEL_NAME, pretrained=True, num_classes=cfg.NUM_CLASSES)

    model = model.to(device)
    return model


def prepare_for_training(model, regularized_blocks=None, all_blocks_capture=False):
    """Prepare model for training with attention weight caching.

    Args:
        model: VisionTransformer
        regularized_blocks: blocks that need differentiable attention weights
        all_blocks_capture: if True, patch all blocks (for analysis)
    """
    if all_blocks_capture:
        block_indices = list(range(len(model.blocks)))
        patch_attention_forward(model, block_indices, differentiable=False)
    elif regularized_blocks:
        patch_attention_forward(model, regularized_blocks, differentiable=True)
    return model


def get_head_mads_for_loss(model, regularized_blocks):
    """Get cached attention weights from regularized blocks.

    Returns:
        dict: {block_idx: attn_weights (B, H, N, N)}
    """
    return get_cached_attn_weights(model, regularized_blocks)


def mask_heads(model, block_idx, head_indices):
    """Zero out specified attention heads in a block's output projection.

    Returns a context manager that restores the original weights.
    """
    attn = model.blocks[block_idx].attn
    proj = attn.proj  # Linear(attn_dim, dim)
    head_dim = attn.head_dim

    class HeadMask:
        def __init__(self):
            self.original_weight = proj.weight.data.clone()
            self.original_bias = proj.bias.data.clone() if proj.bias is not None else None

        def __enter__(self):
            for h in head_indices:
                start = h * head_dim
                end = (h + 1) * head_dim
                proj.weight.data[:, start:end] = 0
            return self

        def __exit__(self, *args):
            proj.weight.data.copy_(self.original_weight)
            if self.original_bias is not None and proj.bias is not None:
                proj.bias.data.copy_(self.original_bias)

    return HeadMask()
