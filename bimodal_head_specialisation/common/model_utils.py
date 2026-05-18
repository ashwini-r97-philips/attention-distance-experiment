"""Model utilities: load ViT-S/16, head masking."""

import os
import torch
import timm


def load_vit_small(cfg, pretrained=True):
    """Load ViT-S/16 via timm, with ImageNet-1K pretrained weights."""
    model = timm.create_model(
        cfg.model_name,
        pretrained=pretrained,
        num_classes=cfg.num_classes,
        img_size=getattr(cfg, "img_size", 224),
    )
    model = model.to(cfg.device)
    return model


def mask_heads(model, block_idx, head_indices):
    """Zero out specified attention heads in a block's output projection.

    Returns a context manager that restores the original weights.
    """
    attn = model.blocks[block_idx].attn
    proj = attn.proj
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
