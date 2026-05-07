# Bimodal Head Specialization in Vision Transformers

**Research question:** Can a pretrained DeiT-S learn stable local and global attention head roles in its early blocks through a training-time distance regularizer, without modifying inference-time architecture?

## Overview

This project investigates whether attention heads in Vision Transformers can be encouraged to specialize along a spatial-distance axis — some heads becoming "local" (attending to nearby patches) and others "global" (attending far away) — via a simple regularizer on Mean Attention Distance (MAD). The hypothesis is that such bimodal specialization improves functional diversity and is particularly useful for dense prediction tasks like segmentation.

## Project Structure

```
attention-distance/
├── config.py                 # Classification experiment config (Tiny-ImageNet, DeiT-S)
├── seg_config.py             # Segmentation experiment config (ADE20K, 512×512)
├── data.py                   # Tiny-ImageNet / ImageFolder data loading
├── seg_data.py               # ADE20K semantic segmentation data loading
├── model_utils.py            # DeiT-S loading, head masking utilities
├── seg_model.py              # DeiT-S encoder + linear segmentation decoder
├── attention_hooks.py        # Monkey-patching timm Attention to extract/cache weights
├── mad_metrics.py            # MAD, non-self MAD, local mass, entropy, distance histograms
├── bimodal_loss.py           # Gap + compactness regularizer on headwise MAD
├── boundary_utils.py         # Boundary extraction, conditional MAD, mIoU, boundary F1
├── baseline_analysis.py      # Analyze pretrained model attention patterns
├── train.py                  # Classification training loop (baseline / regularized)
├── seg_train.py              # Segmentation training loop (baseline / regularized)
├── evaluate.py               # Classification evaluation (GMM, persistence, masking)
├── seg_evaluate.py           # Segmentation evaluation (+ boundary metrics, head norms)
├── generate_report.py        # Classification report generator
├── seg_report.py             # Segmentation report generator
├── plot_utils.py             # All visualization functions
├── run_experiment.sh         # Master script: classification pipeline
├── run_seg_experiment.sh     # Master script: segmentation pipeline
├── download_ade20k.sh        # Download ADE20K dataset
├── requirements.txt          # Python dependencies
├── deep_research.md          # Literature review and novelty analysis
├── first-plan.md             # Detailed experiment plan
├── outputs/                  # Classification experiment results (gitignored)
│   ├── baseline_ft/          # 30-epoch baseline finetuning
│   ├── regularized_ft/       # 30-epoch regularized finetuning
│   ├── analysis/             # Evaluation JSON
│   └── report/               # Generated report + figures
└── seg_outputs/              # Segmentation experiment results (gitignored)
    └── baseline_seg/         # 12-epoch baseline (incomplete)
```

## Environment

- **Python 3.10** (conda env: `jepa-medsam`)
- **PyTorch 2.5.1+cu121**, **timm 1.0.26**
- **Hardware:** NVIDIA L40 (48GB), single GPU via `CUDA_VISIBLE_DEVICES=2`
- **Pretrained weights:** DeiT-S from Facebook CDN (`deit_small_patch16_224-cd65a155.pth`)

## Experiments

### Experiment 1: Classification on Tiny-ImageNet (COMPLETED)

**Setup:**
- DeiT-S (22M params, 12 blocks, 6 heads, embed_dim=384)
- Tiny-ImageNet-200 (100K train / 10K val, 200 classes, 64×64 upscaled to 224×224)
- 30 epochs finetuning from pretrained ImageNet weights
- Regularizer: λ_gap=0.1, λ_compact=0.01, δ=0.3, warmup=5 epochs, blocks 0-5

**Results:**

| Model | Val Top-1 | Mask Local | Mask Global | Mask Random |
|-------|----------:|----------:|----------:|----------:|
| Baseline | 86.38% | 9.36% | 11.98% | 38.21% |
| Regularized | 86.43% | 8.88% | 14.53% | 37.40% |

| Block | Baseline MAD Range | Regularized MAD Range | GMM Winner |
|-------|-------------------:|---------------------:|-----------|
| 0 | 0.3034 | 0.3051 | Baseline |
| 1 | 0.2632 | 0.2622 | Baseline |
| 2 | 0.1753 | 0.1764 | Regularized |
| 3 | 0.1613 | 0.1625 | Regularized |
| 4 | 0.2447 | 0.2477 | Baseline |
| 5 | 0.2382 | 0.2338 | Regularized |

**Role Persistence:** 100% for both models across all blocks.

**Conclusion:** The regularizer had minimal effect. MAD ranges barely moved, and head roles were already perfectly stable in the baseline. Tiny-ImageNet is too easy/small to reveal meaningful specialization differences. The task doesn't require diverse spatial scales.

### Experiment 2: Segmentation on ADE20K (IN PROGRESS — stalled)

**Setup:**
- DeiT-S encoder (512×512 input → 32×32 = 1024 tokens) + linear segmentation decoder
- ADE20K (20K train / 2K val, 150 semantic classes)
- Auxiliary head from block 8
- Stronger regularizer: λ_gap=1.0, λ_compact=0.1, δ=0.3, warmup=10 epochs
- Differential LR: backbone=1e-5, decoder=1e-4
- Position embedding interpolation: 14×14 → 32×32 (bicubic)

**Baseline Progress (12/30 epochs completed, then training stalled):**

| Epoch | mIoU | Boundary F1 | Train Loss |
|------:|-----:|----------:|----------:|
| 1 | 0.1979 | 0.2113 | 2.0985 |
| 5 | 0.3395 | 0.2361 | 1.1011 |
| 10 | 0.3673 | 0.2449 | 0.9045 |
| 12 | 0.3747 | 0.2486 | 0.8503 |

**Status:** Baseline training ran 12 epochs before process instability (DataLoader deadlocks with multiprocessing). Fixed by setting `num_workers=0`. Regularized phase not yet started.

## Key Technical Details

### Attention Hook Mechanism
timm's `Attention` class uses fused SDPA by default, which doesn't expose attention weights. We monkey-patch `Attention.forward` on targeted blocks to:
1. Compute `attn = softmax(Q @ K^T / sqrt(d))`
2. Cache the weights (detached for eval, with gradients for training)
3. Proceed with standard V projection

### Bimodal Loss
For each regularized block, heads are sorted by MAD (using detached values for index selection). Bottom 3 = "local", top 3 = "global":
- **Gap loss:** `max(0, δ - (mean_global - mean_local))` — push groups apart
- **Compactness loss:** `Var(local) + Var(global)` — make each group tight
- **Total:** `warmup_factor × (λ_gap × L_gap + λ_compact × L_compact)`

### Segmentation Decoder
Simple linear decoder: reshape patch tokens to spatial grid → 1×1 Conv+BN+ReLU → 1×1 Conv(→150 classes) → bilinear upsample to original resolution.

## How to Run

```bash
# Activate environment
source ~/miniforge3/etc/profile.d/conda.sh && conda activate jepa-medsam

# Classification experiment (already completed)
CUDA_VISIBLE_DEVICES=2 bash run_experiment.sh

# Segmentation experiment
# 1. Download ADE20K (if not present)
bash download_ade20k.sh /sudarshana/data/

# 2. Run baseline
CUDA_VISIBLE_DEVICES=2 python seg_train.py --mode baseline --epochs 30

# 3. Run regularized
CUDA_VISIBLE_DEVICES=2 python seg_train.py --mode regularized --epochs 30

# 4. Evaluate both
CUDA_VISIBLE_DEVICES=2 python seg_evaluate.py \
    --baseline_ckpt seg_outputs/baseline_seg/checkpoints/best.pth \
    --regularized_ckpt seg_outputs/regularized_seg/checkpoints/best.pth

# 5. Generate report
python seg_report.py --eval_results seg_outputs/analysis/seg_evaluation_results.json
```

## Lessons Learned

1. **Tiny-ImageNet is insufficient** for this research question — heads already stabilize without any regularizer, and the task doesn't demand diverse spatial scales.
2. **DataLoader multiprocessing deadlocks** when processes are killed/restarted. Use `num_workers=0` for reliability (slower but stable) or use `forkserver` context with careful process management.
3. **Dense prediction (segmentation) is the right testbed** — local vs global head specialization should matter functionally when the model needs both boundary precision and global context.
4. **Position embedding interpolation** (14×14→32×32 bicubic) works correctly for DeiT-S at 512×512 resolution.

## Next Steps

### Immediate (Complete Experiment 2)

1. **Resume baseline segmentation training** from epoch 12 → 30 epochs
2. **Run regularized segmentation training** (30 epochs, λ_gap=1.0, λ_compact=0.1)
3. **Full evaluation** comparing both:
   - mIoU and boundary F1
   - GMM bimodality on headwise MAD
   - Role persistence across images
   - **Boundary-conditional MAD** (do heads specialize differently at boundary vs interior tokens?)
   - **Head masking → mIoU + boundary F1** (which roles matter more for which metric?)
   - Distance histograms per head
   - Head output norms (verify no dead heads)

### If Segmentation Shows Clear Specialization

4. **Ablation studies:**
   - Regularizer strength sweep (0.1× to 10× current lambdas)
   - Block selection (first 3 vs first 6 vs all 12)
   - Unbalanced split (2 local / 4 global)
5. **Additional dense tasks:** Cityscapes, COCO panoptic
6. **Robustness testing:** ImageNet-C corruptions (blur, noise, scale)

### Longer-term Research Direction

7. **Token-dependent locality** — let each token learn its own distance law (highest novelty idea)
8. **Semantic alignment** — attention distance conditioned on semantic boundaries
9. **Paper positioning:** "Attention distance is a learnable structured variable" — distinct from windowed/sparse approaches (DAT, VSA, HiLo, LocAtViT)

## References

Key prior work this builds on:
- Raghu et al. (2021) — "Do Vision Transformers See Like CNNs?" (attention distance measurements)
- d'Ascoli et al. (2021) — ConViT (soft locality that heads can escape)
- Wang et al. (2022) — HiLo (hard local/global head partition)
- Chen et al. (2022) — Principle of Diversity (head redundancy reduction)
- LocAtViT (ICLR 2026) — learnable Gaussian locality bias
