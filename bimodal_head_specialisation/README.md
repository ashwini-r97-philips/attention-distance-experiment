# Bimodal Head Specialisation in Vision Transformers

**Research question:** Can a pretrained ViT-S/16 learn stable local and global attention head roles through a training-time distance regulariser, without modifying inference-time architecture?

## Overview

This experiment investigates whether attention heads in Vision Transformers can be encouraged to specialise along a spatial-distance axis — some heads becoming "local" (attending to nearby patches) and others "global" (attending far away) — via regularisers on Mean Attention Distance (MAD). Two regulariser types are supported:

1. **Spread loss**: maximise inter-head MAD variance per layer (`-Var_h(d_lh)`)
2. **Bimodal mixture loss**: encourage each head's MAD to fit a two-component Gaussian prior

## Structure

```
bimodal_head_specialisation/
├── README.md                 # This file
├── common/                   # Shared modules
│   ├── config.py             # YAML-based configuration system
│   ├── attention_hooks.py    # Monkey-patching timm Attention to extract/cache weights
│   ├── mad_metrics.py        # MAD, local mass, entropy, distance histograms, head correlation
│   ├── model_utils.py        # ViT-S/16 loading, head masking
│   ├── regularisers.py       # SpreadLoss and BimodalMixtureLoss
│   └── plot_utils.py         # All visualisation functions
├── classification/           # ImageNet-1K classification
│   ├── configs/              # YAML config files
│   │   ├── baseline.yaml
│   │   ├── spread_weak.yaml
│   │   └── bimodal_weak.yaml
│   ├── data.py               # ImageNet-1K data loading
│   ├── train.py              # Training loop
│   ├── evaluate.py           # Evaluation (accuracy, attention stats, GMM)
│   ├── visualize.py          # Generate all plots
│   └── run_experiment.sh     # Master pipeline script
├── segmentation/             # ADE20K segmentation (separate)
│   └── ...
└── runs/                     # Experiment outputs (gitignored)
    ├── baseline_vit_s16_imagenet1k/
    │   ├── config.yaml
    │   ├── checkpoints/
    │   ├── training_log.json
    │   ├── attention_stats/
    │   └── eval/
    └── spread_weak_vit_s16_imagenet1k/
        └── ...
```

## Model

- **ViT-S/16** (vanilla Vision Transformer, 22M params)
- 12 layers, 6 heads, embed_dim=384, patch_size=16
- Input: 224×224 → 14×14 = 196 patch tokens + CLS
- Pretrained weights from timm (ImageNet-1K supervised)

## How to Run

### Classification (ImageNet-1K)

```bash
cd classification/

# Baseline
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/baseline.yaml

# Spread regulariser
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/spread_weak.yaml

# Bimodal regulariser
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/bimodal_weak.yaml

# Evaluate
python evaluate.py --config configs/baseline.yaml \
    --checkpoint runs/baseline_vit_s16_imagenet1k/checkpoints/best.pth

# Visualise (compare baseline vs targeted)
python visualize.py \
    --baseline_results runs/baseline_vit_s16_imagenet1k/eval/evaluation_results.json \
    --targeted_results runs/spread_weak_vit_s16_imagenet1k/eval/evaluation_results.json \
    --baseline_log runs/baseline_vit_s16_imagenet1k/training_log.json \
    --targeted_log runs/spread_weak_vit_s16_imagenet1k/training_log.json

# Or run full pipeline:
CUDA_VISIBLE_DEVICES=0 bash run_experiment.sh
```

## Metrics & Visualisations

### Training metrics logged per epoch
- Train loss, val loss, top-1, top-5, learning rate, gradient norm, GPU memory
- Regulariser loss (separate from CE)
- Per-layer/per-head MAD values

### Attention analysis (on fixed val subset)
- Mean Attention Distance (MAD) per layer × head
- Local mass at τ = 0.15, 0.25, 0.35
- Attention entropy
- Inter-head MAD variance
- Distance histograms
- Head correlation matrices

### Generated plots
1. Training curves (loss, accuracy, LR, reg loss)
2. MAD heatmaps (baseline, targeted, difference)
3. MAD distributions by layer
4. Headwise MAD trajectories over training
5. Inter-head MAD variance by layer
6. Local mass heatmaps per τ
7. Entropy heatmaps
8. Attention distance histograms
9. Bimodality diagnostics (histogram + GMM AIC/BIC)
10. Summary table (CSV + markdown)

## Fairness Requirements

All runs use identical:
- Random seed, data split, preprocessing
- Training schedule (cosine LR with warmup)
- Augmentation (RandAugment + Mixup/CutMix)
- Batch size, optimizer (AdamW), weight decay
- Git commit hash logged per run
