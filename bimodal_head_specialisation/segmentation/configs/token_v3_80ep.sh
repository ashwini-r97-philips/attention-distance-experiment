#!/usr/bin/env bash
# Condition 3: Content-conditioned per-head signed locality gate (v3).
# Each token predicts its own local/global bias per head via Linear(384→6).
# Gate params: 6 blocks * (384*6 + 6) bias = ~13,860 params.
# Gate LR = backbone_lr * gate_lr_multiplier = 1e-5 * 10 = 1e-4.
#
# gate_scale=2.0    → gate values in (-2, +2); positive=local, negative=global
# gate_distance_scale=2.0 → multiplier on distance penalty before softmax
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python train.py \
    --mode token_locality_v3 \
    --epochs 80 \
    --batch_size 8 \
    --backbone_lr 1e-5 \
    --decoder_lr 1e-4 \
    --gate_blocks 0 1 2 3 4 5 \
    --gate_lr_multiplier 10.0 \
    --gate_scale 2.0 \
    --gate_distance_scale 2.0 \
    --gate_weight_std 0.02 \
    --output_dir runs/seg/token_v3_80ep \
    "$@"
