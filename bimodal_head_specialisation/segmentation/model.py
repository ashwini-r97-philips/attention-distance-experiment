"""DeiT-S encoder + linear segmentation decoder.

Uses timm's VisionTransformer with img_size=512, producing 32×32 = 1024 patch tokens.
Simple linear decoder: reshape patch tokens → 1×1 conv → bilinear upsample.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from common.model_utils import DEIT_S_CACHE


class LinearSegDecoder(nn.Module):
    """Simple linear segmentation head.

    Takes (B, N_patches, embed_dim) → reshape to (B, C, H, W) → 1×1 conv → upsample.
    """

    def __init__(self, embed_dim, num_classes, grid_h, grid_w):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, num_classes, kernel_size=1),
        )

    def forward(self, patch_tokens, output_size=None):
        """
        Args:
            patch_tokens: (B, N_patches, embed_dim) — CLS token already removed
            output_size: (H, W) target spatial size for upsampling
        """
        B, N, C = patch_tokens.shape
        x = patch_tokens.transpose(1, 2).reshape(B, C, self.grid_h, self.grid_w)
        x = self.head(x)  # (B, num_classes, grid_h, grid_w)
        if output_size is not None:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x


class DeiTSegModel(nn.Module):
    """DeiT-S encoder + linear segmentation decoder.

    The encoder runs at img_size=512 (32×32 patches). Position embeddings are
    interpolated automatically by timm when img_size differs from pretrained 224.
    """

    def __init__(self, num_classes=None, pretrained=True):
        super().__init__()
        num_classes = num_classes or cfg.NUM_SEG_CLASSES

        # Create encoder with 512×512 input — timm handles pos embed interpolation
        self.encoder = timm.create_model(
            cfg.MODEL_NAME,
            pretrained=False,
            img_size=cfg.IMG_SIZE,
            num_classes=0,       # remove classification head
            global_pool="",      # no pooling — we want all patch tokens
        )

        # Load pretrained weights manually (pos_embed will be interpolated)
        if pretrained:
            self._load_pretrained()

        self.decoder = LinearSegDecoder(
            embed_dim=cfg.EMBED_DIM,
            num_classes=num_classes,
            grid_h=cfg.GRID_H,
            grid_w=cfg.GRID_W,
        )

        # Auxiliary head from block 8 (optional, helps training)
        self.aux_head = LinearSegDecoder(
            embed_dim=cfg.EMBED_DIM,
            num_classes=num_classes,
            grid_h=cfg.GRID_H,
            grid_w=cfg.GRID_W,
        )
        self.aux_block_idx = 8

    def _load_pretrained(self):
        """Load Facebook DeiT-S weights with pos_embed interpolation."""
        if os.path.exists(DEIT_S_CACHE):
            ckpt = torch.load(DEIT_S_CACHE, map_location="cpu", weights_only=True)
            state_dict = ckpt["model"] if "model" in ckpt else ckpt

            # Filter to matching keys, allow shape mismatch for pos_embed (will interpolate)
            model_sd = self.encoder.state_dict()
            filtered = {}
            for k, v in state_dict.items():
                if k not in model_sd:
                    continue
                if k == "pos_embed":
                    # Interpolate from (1, 197, 384) to (1, 1025, 384)
                    v = self._interpolate_pos_embed(v, model_sd[k].shape)
                    filtered[k] = v
                elif v.shape == model_sd[k].shape:
                    filtered[k] = v

            missing = set(model_sd.keys()) - set(filtered.keys())
            self.encoder.load_state_dict(filtered, strict=False)
            if missing:
                print(f"  Encoder: {len(filtered)} loaded, {len(missing)} missing (init): "
                      f"{[k for k in sorted(missing) if 'head' not in k][:5]}...")
        else:
            print("WARNING: No cached DeiT-S weights found. Using random init.")

    @staticmethod
    def _interpolate_pos_embed(pos_embed, target_shape):
        """Interpolate position embedding from 14×14+1 to 32×32+1."""
        # pos_embed: (1, 197, D) — 1 CLS + 196 patch tokens (14×14)
        # target:    (1, 1025, D) — 1 CLS + 1024 patch tokens (32×32)
        cls_token = pos_embed[:, :1, :]  # (1, 1, D)
        patch_embed = pos_embed[:, 1:, :]  # (1, 196, D)

        D = patch_embed.shape[-1]
        old_grid = int(patch_embed.shape[1] ** 0.5)  # 14

        target_num_patches = target_shape[1] - 1  # 1024
        new_grid = int(target_num_patches ** 0.5)  # 32

        patch_embed = patch_embed.reshape(1, old_grid, old_grid, D).permute(0, 3, 1, 2)
        patch_embed = F.interpolate(patch_embed, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
        patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, D)

        return torch.cat([cls_token, patch_embed], dim=1)

    @property
    def blocks(self):
        """Access encoder blocks directly (for attention hooks)."""
        return self.encoder.blocks

    def forward(self, x, return_aux=True):
        """Forward pass.

        Returns:
            logits: (B, num_classes, H, W) at original resolution
            aux_logits: (B, num_classes, H, W) from intermediate block (if return_aux)
        """
        B, C, H, W = x.shape

        # Forward through encoder, capturing intermediate for aux head
        # We manually iterate blocks to grab the intermediate feature
        x_enc = self.encoder.patch_embed(x)
        x_enc = self.encoder._pos_embed(x_enc)
        x_enc = self.encoder.patch_drop(x_enc)
        x_enc = self.encoder.norm_pre(x_enc)

        aux_features = None
        for i, blk in enumerate(self.encoder.blocks):
            x_enc = blk(x_enc)
            if i == self.aux_block_idx:
                aux_features = self.encoder.norm(x_enc)

        x_enc = self.encoder.norm(x_enc)

        # Remove CLS token — patch tokens only
        num_prefix = self.encoder.num_prefix_tokens  # usually 1 (CLS)
        patch_tokens = x_enc[:, num_prefix:, :]  # (B, N_patches, D)

        logits = self.decoder(patch_tokens, output_size=(H, W))

        if return_aux and aux_features is not None:
            aux_patch = aux_features[:, num_prefix:, :]
            aux_logits = self.aux_head(aux_patch, output_size=(H, W))
            return logits, aux_logits

        return logits

    def get_encoder_param_groups(self, backbone_lr, decoder_lr, gate_module=None, gate_lr=None):
        """Separate param groups for differential learning rates.

        If gate_module is provided, its parameters are placed in a third group
        at gate_lr (defaults to 10× backbone_lr if not specified).
        """
        gate_param_ids = set()
        gate_params = []
        if gate_module is not None:
            gate_params = gate_module.get_gate_params()
            gate_param_ids = {id(p) for p in gate_params}

        encoder_params = [p for p in self.encoder.parameters() if id(p) not in gate_param_ids]
        decoder_params = list(self.decoder.parameters()) + list(self.aux_head.parameters())

        groups = [
            {"params": encoder_params, "lr": backbone_lr},
            {"params": decoder_params, "lr": decoder_lr},
        ]
        if gate_params:
            lr = gate_lr if gate_lr is not None else backbone_lr * 10.0
            groups.append({"params": gate_params, "lr": lr})
        return groups
