# Bimodal Head Specialisation in Vision Transformers

**Research question:** Can a pretrained DeiT-S learn stable local and global attention head roles in its early blocks through a training-time distance regulariser, without modifying inference-time architecture?

## Overview

This experiment investigates whether attention heads in Vision Transformers can be encouraged to specialise along a spatial-distance axis — some heads becoming "local" (attending to nearby patches) and others "global" (attending far away) — via a simple regulariser on Mean Attention Distance (MAD). The hypothesis is that such bimodal specialisation improves functional diversity and is particularly useful for dense prediction tasks like segmentation.

## Structure

```
bimodal_head_specialisation/
├── README.md                 # This file
├── common/                   # Shared modules used by both tasks
│   ├── config.py             # Base configuration (DeiT-S model params, regulariser)
│   ├── attention_hooks.py    # Monkey-patching timm Attention to extract/cache weights
│   ├── mad_metrics.py        # MAD, non-self MAD, local mass, entropy, distance histograms
│   ├── model_utils.py        # DeiT-S loading, head masking utilities
│   ├── bimodal_loss.py       # Gap + compactness regulariser on headwise MAD
│   ├── boundary_utils.py     # Boundary extraction, conditional MAD, mIoU, boundary F1
│   └── plot_utils.py         # All visualisation functions
├── classification/           # Classification on Tiny-ImageNet
│   ├── data.py               # Tiny-ImageNet / ImageFolder data loading
│   ├── train.py              # Training loop (baseline / regularised)
│   ├── baseline_analysis.py  # Analyse pretrained model attention patterns
│   ├── evaluate.py           # Evaluation (GMM, persistence, masking)
│   ├── generate_report.py    # Report generator
│   └── run_experiment.sh     # Master pipeline script
├── segmentation/             # Segmentation on ADE20K
│   ├── config.py             # Segmentation-specific config (overrides common)
│   ├── data.py               # ADE20K data loading
│   ├── model.py              # DeiT-S encoder + linear segmentation decoder
│   ├── train.py              # Training loop (baseline / regularised)
│   ├── evaluate.py           # Evaluation (+ boundary metrics, head norms)
│   ├── report.py             # Report generator
│   ├── download_ade20k.sh    # Download ADE20K dataset
│   └── run_experiment.sh     # Master pipeline script
├── outputs/                  # Classification results (gitignored)
└── seg_outputs/              # Segmentation results (gitignored)
```

## Key Technical Details

### Attention Hook Mechanism
timm's `Attention` class uses fused SDPA by default, which doesn't expose attention weights. We monkey-patch `Attention.forward` on targeted blocks to:
1. Compute `attn = softmax(Q @ K^T / sqrt(d))`
2. Cache the weights (detached for eval, with gradients for training)
3. Proceed with standard V projection

### Bimodal Loss
For each regularised block, heads are sorted by MAD (using detached values for index selection). Bottom 3 = "local", top 3 = "global":
- **Gap loss:** `max(0, δ - (mean_global - mean_local))` — push groups apart
- **Compactness loss:** `Var(local) + Var(global)` — make each group tight
- **Total:** `warmup_factor × (λ_gap × L_gap + λ_compact × L_compact)`

### Segmentation Decoder
Simple linear decoder: reshape patch tokens to spatial grid → 1×1 Conv+BN+ReLU → 1×1 Conv(→150 classes) → bilinear upsample to original resolution.

## How to Run

### Classification (Tiny-ImageNet)

```bash
cd classification/
CUDA_VISIBLE_DEVICES=2 bash run_experiment.sh
```

Or individual phases:
```bash
cd classification/
CUDA_VISIBLE_DEVICES=2 python train.py --mode baseline --epochs 30
CUDA_VISIBLE_DEVICES=2 python train.py --mode regularized --epochs 30
CUDA_VISIBLE_DEVICES=2 python evaluate.py \
    --baseline_ckpt ../outputs/baseline_ft/checkpoints/best.pth \
    --regularized_ckpt ../outputs/regularized_ft/checkpoints/best.pth
python generate_report.py --eval_results ../outputs/analysis/evaluation_results.json
```

### Segmentation (ADE20K)

```bash
cd segmentation/
CUDA_VISIBLE_DEVICES=2 bash run_experiment.sh
```

Or individual phases:
```bash
cd segmentation/
bash download_ade20k.sh /sudarshana/data/
CUDA_VISIBLE_DEVICES=2 python train.py --mode baseline --epochs 30
CUDA_VISIBLE_DEVICES=2 python train.py --mode regularized --epochs 30
CUDA_VISIBLE_DEVICES=2 python evaluate.py \
    --baseline_ckpt ../seg_outputs/baseline_seg/checkpoints/best.pth \
    --regularized_ckpt ../seg_outputs/regularized_seg/checkpoints/best.pth
python report.py --eval_results ../seg_outputs/analysis/seg_evaluation_results.json
```

## Results

### Classification (Tiny-ImageNet) — COMPLETED

| Model | Val Top-1 | Mask Local | Mask Global | Mask Random |
|-------|----------:|----------:|----------:|----------:|
| Baseline | 86.38% | 9.36% | 11.98% | 38.21% |
| Regularised | 86.43% | 8.88% | 14.53% | 37.40% |

**Conclusion:** The regulariser had minimal effect on Tiny-ImageNet. Head roles were already perfectly stable in the baseline. The task is too easy/small to reveal meaningful specialisation differences.

### Segmentation (ADE20K) — IN PROGRESS

Baseline training reached 12/30 epochs (mIoU 0.3747, boundary F1 0.2486) before process instability. Regularised phase not yet started.
