# Attention Distance Experiments

Research on attention distance properties in Vision Transformers — investigating how spatial attention patterns can be measured, structured, and leveraged for improved performance.

## Project Structure

```
attention-distance-experiment/
├── README.md                          # This file
├── requirements.txt                   # Shared Python dependencies
└── bimodal_head_specialisation/       # Experiment 1: Bimodal head roles via MAD regulariser
    ├── README.md                      # Experiment details, results, how to run
    ├── common/                        # Shared modules (hooks, metrics, model utils, loss, plots)
    ├── classification/                # Tiny-ImageNet classification task
    └── segmentation/                  # ADE20K semantic segmentation task
```

## Experiments

### 1. Bimodal Head Specialisation (active)

Can attention heads be trained to specialise as "local" or "global" via a distance regulariser on Mean Attention Distance? Tested on classification (Tiny-ImageNet) and segmentation (ADE20K).

See [bimodal_head_specialisation/README.md](bimodal_head_specialisation/README.md) for details.

## Environment

- **Python 3.10** (conda env: `jepa-medsam`)
- **PyTorch 2.5.1+cu121**, **timm 1.0.26**
- **Hardware:** NVIDIA L40 (48GB), single GPU via `CUDA_VISIBLE_DEVICES=2`

## References

Key prior work:
- Raghu et al. (2021) — "Do Vision Transformers See Like CNNs?" (attention distance measurements)
- d'Ascoli et al. (2021) — ConViT (soft locality that heads can escape)
- Wang et al. (2022) — HiLo (hard local/global head partition)
- Chen et al. (2022) — Principle of Diversity (head redundancy reduction)
- LocAtViT (ICLR 2026) — learnable Gaussian locality bias
