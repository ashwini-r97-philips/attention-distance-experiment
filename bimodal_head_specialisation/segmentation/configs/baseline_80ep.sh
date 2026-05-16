#!/usr/bin/env bash
# Condition 1: DeiT-S + linear decoder, no locality bias.
# 80 epochs — same budget as gaussian_bias and token_v3 runs for fair comparison.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python train.py \
    --mode baseline \
    --epochs 80 \
    --batch_size 8 \
    --backbone_lr 1e-5 \
    --decoder_lr 1e-4 \
    --output_dir runs/seg/baseline_80ep \
    "$@"
