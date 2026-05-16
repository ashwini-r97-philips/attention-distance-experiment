#!/usr/bin/env bash
# Condition 2: LocAtViT-style per-head Gaussian spatial bias (no content conditioning).
# Ablation baseline — isolates whether spatial prior alone helps vs v3's content conditioning.
#
# Gate params: 2 * num_heads * num_blocks = 2 * 6 * 6 = 72 scalars total.
# Gate LR = backbone_lr * gate_lr_multiplier = 1e-5 * 10 = 1e-4.
# init_sigma=0.25 → Gaussian width ≈ 25% of max patch distance at start.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python train.py \
    --mode gaussian_bias \
    --epochs 80 \
    --batch_size 8 \
    --backbone_lr 1e-5 \
    --decoder_lr 1e-4 \
    --gate_blocks 0 1 2 3 4 5 \
    --gate_lr_multiplier 10.0 \
    --gaussian_init_sigma 0.25 \
    --output_dir runs/seg/gaussian_bias_80ep \
    "$@"
