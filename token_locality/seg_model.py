"""DeiT-S encoder + linear segmentation decoder for ADE20K.

Copied and made self-contained from bimodal_head_specialisation/segmentation/model.py.
Uses timm's built-in pretrained weight download (no external cache dependency).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ─── Constants (mirror segmentation/config.py) ───────────────────────────────
MODEL_NAME   = "deit_small_patch16_224"
NUM_CLASSES  = 150
IGNORE_INDEX = 255
EMBED_DIM    = 384
NUM_HEADS    = 6
NUM_BLOCKS   = 12
PATCH_SIZE   = 16
IMG_SIZE     = 512
GRID_H       = IMG_SIZE // PATCH_SIZE   # 32
GRID_W       = IMG_SIZE // PATCH_SIZE   # 32
NUM_PATCHES  = GRID_H * GRID_W          # 1024


class LinearSegDecoder(nn.Module):
    """1×1 conv head: (B, N_patches, D) → (B, num_classes, H, W)."""

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
        B, N, C = patch_tokens.shape
        x = patch_tokens.transpose(1, 2).reshape(B, C, self.grid_h, self.grid_w)
        x = self.head(x)
        if output_size is not None:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x


class DeiTSegModel(nn.Module):
    """DeiT-S encoder + linear decoder + auxiliary head from block 8.

    pretrained=True uses timm's automatic weight download.
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()

        self.encoder = timm.create_model(
            MODEL_NAME,
            pretrained=pretrained,
            img_size=IMG_SIZE,
            num_classes=0,
            global_pool="",
        )

        self.decoder = LinearSegDecoder(EMBED_DIM, num_classes, GRID_H, GRID_W)
        self.aux_head = LinearSegDecoder(EMBED_DIM, num_classes, GRID_H, GRID_W)
        self.aux_block_idx = 8

    @property
    def blocks(self):
        return self.encoder.blocks

    def forward(self, x, return_aux=True):
        B, C, H, W = x.shape

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

        num_prefix = self.encoder.num_prefix_tokens
        patch_tokens = x_enc[:, num_prefix:, :]
        logits = self.decoder(patch_tokens, output_size=(H, W))

        if return_aux and aux_features is not None:
            aux_patch = aux_features[:, num_prefix:, :]
            aux_logits = self.aux_head(aux_patch, output_size=(H, W))
            return logits, aux_logits

        return logits

    def get_encoder_param_groups(self, backbone_lr, decoder_lr,
                                 gate_module=None, gate_lr=None):
        gate_param_ids = set()
        gate_params = []
        if gate_module is not None:
            gate_params = gate_module.get_gate_params()
            gate_param_ids = {id(p) for p in gate_params}

        encoder_params = [p for p in self.encoder.parameters()
                          if id(p) not in gate_param_ids]
        decoder_params = (list(self.decoder.parameters())
                          + list(self.aux_head.parameters()))

        groups = [
            {"params": encoder_params, "lr": backbone_lr},
            {"params": decoder_params, "lr": decoder_lr},
        ]
        if gate_params:
            groups.append({"params": gate_params,
                           "lr": gate_lr if gate_lr is not None else backbone_lr * 10.0})
        return groups
