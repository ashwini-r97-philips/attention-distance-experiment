# Token Locality Experiment

This folder contains a self-contained experiment testing **learnable locality gates** injected directly into ViT-S/16 attention logits. The core question: can a small learned module, operating inside each attention block, teach individual heads to become local or global on a per-token basis — and does this improve fine-grained recognition or semantic segmentation?

---

## Conceptual Background

### What the gate does

Standard ViT attention computes:

```
attention[b, h, i, j] = softmax( (Q_i · K_j) / sqrt(d) )
```

The locality gate intercepts this *before* softmax and injects a spatially-structured penalty:

```
attention[b, h, i, j] = softmax( (Q_i · K_j) / sqrt(d)  −  gate[b, i, h] × dist(i, j) × scale )
```

Where `dist(i, j)` is the normalised Euclidean distance between patch positions `i` and `j` on the grid, and `gate[b, i, h]` is a per-token, per-head scalar predicted from the token's own features.

- **Positive gate → local bias**: distant patches are suppressed, nearby patches are emphasised.
- **Negative gate → global bias**: distant patches are *boosted*, nearby patches suppressed.
- **Gate ≈ 0 → no change**: model behaves exactly like the baseline.

The gate is a tiny branch (a single linear layer `Linear(embed_dim, num_heads)`) attached to the normed input of each attention block. It adds roughly 2,310 parameters per block (384×6 + 6 bias). Training cost is negligible.

### Why this is different from external regularisers

The bimodal regulariser (in `bimodal_head_specialisation/`) is a loss term that *encourages* heads to have different MAD values via gradient descent on Q and K weights. It acts indirectly and weakly. The gate acts *directly inside the attention computation* before the softmax — it structurally rewires which tokens can attend to each other on a per-image, per-token basis. This is a fundamentally stronger and more targeted intervention.

---

## Gate Versions

### v1 — `token_locality_gate.py`

```
Linear(embed_dim → 1)  →  Softplus  →  scalar gate per token
```

- Single scalar gate shared across all heads.
- Softplus ensures gate ≥ 0 (local-only forcing, no global).
- **Critical flaw**: Softplus(0) = ln(2) ≈ 0.693. With zero-init weights, the gate starts at 0.693 regardless of input — a *constant* non-zero penalty from epoch 0. The model never starts as the baseline.
- See `diagnose_gate.py` for a full diagnostic of this failure mode.

### v2 — `token_locality_gate_v2.py`

```
Linear(embed_dim → 1)  →  Softplus  →  scalar gate per token
```

- Identical architecture but fixes v1's init problem with `bias = −5.0` so `Softplus(−5) ≈ 0.007`.
- Gate starts near-zero. Model starts as unmodified baseline.
- Still Softplus (local-only), still scalar (all heads share the same gate value).
- Also includes `FixedLocalityPrior`: a non-learnable constant penalty for ablation.

### v3 — `token_locality_gate_v3.py` ← **current best, use this**

```
Linear(embed_dim → num_heads)  →  gate_scale × tanh(·)  →  (B, N, H) gate per token per head
```

Key improvements over v2:

| Property | v2 | v3 |
|---|---|---|
| Output range | [0, ∞) local-only | (−gate_scale, +gate_scale) bidirectional |
| Per-head | No — all heads share gate | Yes — each head independent |
| Init | bias=−5 hack to fight Softplus | tanh(0)=0 naturally starts as baseline |
| Parameter count (6 blocks) | 6 × (384+1) = 2,310 | 6 × (384×6 + 6) = 13,860 |

The penalty is: `gate[b, i, h] × dist(i, j) × gate_distance_scale` where `gate[b, i, h] ∈ (−2, +2)`.

Also contains `VectorKernelGateBranch` and `VectorKernelGateModule` — a more expressive variant where each token predicts a K-dim vector and the penalty is a dot product with RBF-encoded distance features. **Not wired into training** — code is ready for future integration.

### Segmentation gates — `gate_modules_seg.py`

Contains the LocAtViT-style comparison ablation for ADE20K segmentation:

- `LearnedGaussianBias` / `LearnedGaussianBiasModule`: per-head learnable α_h and σ_h scalars, no content conditioning. Adds `α_h × exp(−dist² / (2σ_h²))` to attention logits. Only 2 × num_heads × num_blocks = 72 parameters total across 6 blocks.

This is the **LocAtViT** comparison condition: isolates whether spatial prior alone helps, before asking whether content conditioning (v3) adds anything.

---

## Datasets

### CUB-200-2011 (fine-grained classification)

Used by `train_cub.py`. Downloaded automatically from HuggingFace — no manual download needed.

```python
# Loaded via:
from datasets import load_dataset
ds = load_dataset("bentrevett/caltech-ucsd-birds-200-2011")
# Train split: 5,994 images across 200 bird species
# Test split:  5,794 images
```

**Requirements:**
- `pip install datasets` (already in requirements.txt)
- Internet access on first run (cached to `~/.cache/huggingface/datasets/` after that)
- No HuggingFace login required — this dataset is public

**What it is:** 200 fine-grained bird species with tight visual differences. A good testbed for locality gates because correctly distinguishing species (e.g. different warbler subspecies) requires attending to specific small parts (bill shape, wing bars, eye rings) — exactly the scenario where a content-conditioned local bias might help.

### ImageNet-1K (full classification + locality)

Used by `train_token_locality.py`. Streamed from HuggingFace — no full download needed but requires authentication.

```bash
huggingface-cli login   # one-time, or set HF_TOKEN env var
```

**What it is:** 1.28M images across 1,000 classes. The `train_token_locality.py` script runs a 5-epoch fine-tuning experiment starting from a pretrained ViT-S/16 checkpoint. Primarily used to verify gate learning dynamics at scale before committing to longer CUB runs.

### ADE20K (semantic segmentation)

Used by `bimodal_head_specialisation/segmentation/train.py` (the segmentation experiment that uses gates from this folder). **Must be downloaded manually.**

```bash
cd bimodal_head_specialisation/segmentation
bash download_ade20k.sh
```

Data location is hardcoded in `bimodal_head_specialisation/segmentation/config.py`:
```python
DATA_ROOT = "/sudarshana/data/ADEChallengeData2016"
```
Change this to match your machine before running.

**What it is:** 20,210 training / 2,000 validation images across 150 semantic classes. Resolution 512×512 for training. The ViT-S/16 encoder produces a 32×32 = 1,024 patch token grid at this resolution. Each block's attention matrix is 1,024×1,024 (vs 196×196 for classification at 224px).

---

## Experiments and Training Scripts

### 1. CUB fine-grained classification

```bash
cd token_locality

# Baseline — no gate, pure fine-tuning
python train_cub.py --config configs/cub/baseline_cub_vit_s16.yaml

# 50-epoch baseline (more converged)
python train_cub.py --config configs/cub/baseline_50ep_cub_vit_s16.yaml

# v3 gate — 50 epochs, per-head signed gate on blocks 0–5
python train_cub.py --config configs/cub/token_v3_50ep_cub_vit_s16.yaml

# Debug run (2 epochs, extra prints)
python train_cub.py --config configs/cub/baseline_cub_vit_s16.yaml --epochs 2 --debug
```

All results written to `runs/cub/<run_name>/` relative to the repo root.

### 2. ImageNet-1K locality (5-epoch gate learning probe)

```bash
cd token_locality
python train_token_locality.py --config configs/token_locality_minimal_vit_s16.yaml
```

This is a quick 5-epoch sanity check that the gate learns on a larger, more varied dataset. It streams ImageNet so it needs HuggingFace login. After training it automatically runs a localization eval on a 2,000-image subset and prints a comparison against the bimodal_medium baseline.

### 3. ADE20K segmentation (3-condition comparison)

```bash
cd bimodal_head_specialisation/segmentation

# Condition 1: DeiT-S baseline (no gate)
bash configs/baseline_80ep.sh

# Condition 2: LocAtViT-style Gaussian prior (spatial bias, no content conditioning)
bash configs/gaussian_bias_80ep.sh

# Condition 3: v3 content-conditioned per-head signed gate
bash configs/token_v3_80ep.sh
```

All three run for 80 epochs with identical backbone LR, decoder LR, and batch size for fair comparison.

---

## Config Fields Reference

All CUB configs are YAML files in `configs/cub/`. The common fields:

```yaml
model_name: vit_small_patch16_224   # timm model name
hf_dataset: bentrevett/caltech-ucsd-birds-200-2011
output_dir: runs/cub/<run_name>
num_classes: 200

# Training
epochs: 50
batch_size: 384          # auto-reduced if OOM
lr: 0.00001              # backbone LR
weight_decay: 0.05
warmup_epochs: 3
label_smoothing: 0.0
seed: 42
num_workers: 4
grad_clip_norm: 1.0

# Gate (only for non-baseline configs)
reg_type: token_locality_v3          # none / token_locality / token_locality_v2 / token_locality_v3 / fixed_locality_prior
regularized_blocks: [0, 1, 2, 3, 4, 5]
gate_distance_scale: 2.0             # multiplier on distance penalty
gate_scale: 2.0                      # tanh ceiling: gate values in (−2, +2)
gate_weight_std: 0.02                # init std for gate linear weights
gate_lr_multiplier: 10.0             # gate params trained at 10× backbone LR

# Attention analysis (logged every epoch)
tau_values: [0.15, 0.25, 0.35]
attention_eval_subset_size: 512      # images used for attention stats eval
```

---

## Output Structure

Each run writes to `runs/<task>/<run_name>/`:

```
runs/cub/token_v3_50ep_cub_vit_s16/
├── config.yaml                  # snapshot of all hyperparameters
├── git_commit.txt               # git hash for reproducibility
├── training_log.csv             # epoch-level: loss, acc1, acc5, lr, time
├── per_layer_head_mad.csv       # per-epoch, per-block, per-head MAD
├── gate_stats.csv               # per-epoch gate weight/output statistics
├── runtime_summary.txt          # best accuracy, peak memory, config summary
├── checkpoints/
│   ├── best.pth                 # best val_acc1 checkpoint
│   └── last.pth                 # most recent epoch checkpoint
├── attention_stats/
│   └── epoch_XXXX.json         # per-block {mad, entropy, local_mass} per epoch
└── gate_stats/
    └── epoch_XXXX.json         # per-block gate weight and output statistics
```

Checkpoints contain:
- `model_state_dict` — full ViT-S/16 weights
- `gate_state_dict` — gate branch weights (only for gated runs)
- `optimizer_state_dict` — for resuming

---

## Diagnosing Gate Behaviour

If you train a v1 or v2 checkpoint and want to understand what the gate learned (or failed to learn):

```bash
cd token_locality
python diagnose_gate.py --checkpoint runs/cub/token_locality_mild_cub_vit_s16/checkpoints/best.pth
```

The diagnostic checks:
1. **Gate weight statistics** — are the weights actually non-zero after training?
2. **Gate output magnitudes** — is the gate varying with input or outputting a constant?
3. **Pre-activation values** — what does `Linear(x)` output before the nonlinearity?
4. **Penalty vs logit scale** — is the penalty large enough to actually shift attention?
5. **Gradient analysis** — are gradients flowing into the gate at all?

This script targets v1/v2 gates. For v3, use the gate statistics logged to `gate_stats/` during training — look at `gate_output_mean` and `gate_output_std` to verify the gate is producing token-varying outputs (std should grow across training; if it stays near 0, the gate is not differentiating tokens).

---

## Key Things to Watch During Training

**For v3 gate on CUB:**
- `gate_output_mean` should drift away from 0 (positive = overall local bias, negative = global)
- `gate_output_std` should grow from ~0 across epochs — this is the signature of the gate learning token-dependent behaviour
- Per-head gate outputs: some heads should go positive, others negative, indicating spontaneous local/global specialisation
- Val acc1 should meet or exceed the 50-epoch baseline; a drop >1% suggests the gate is hurting more than helping

**For segmentation (all three conditions):**
- `val_miou` is the primary metric — compare all three at epoch 80
- `val_boundary_f1` measures edge precision — locality gates should help here specifically
- Gate stats: `alpha_mean` for Gaussian bias (should move away from 0), `gate_output_mean` for v3
- MAD logs (every 5 epochs): do gated blocks show different MAD distributions vs baseline?

---

## Module Interaction

The gate modules do **not** use `capture_attention` from `common/attention_hooks.py`. They install their own monkey-patched forwards directly on `attn_module.forward`. This means:

- You must **not** call `patch_attention_forward` or `unpatch_attention_forward` on gated blocks — it will destroy the gate installation.
- The `compute_attention_stats` function in `train_cub.py` is written to handle this: it only patches *non-gate* blocks temporarily for the eval forward pass, and restores only those.
- `gate_module.remove_gates(model)` safely restores original forwards if you need to run without the gate (used in `diagnose_gate.py`).

---

## Extending This Code

### Wire in the VectorKernelGate (future)

The `VectorKernelGateBranch` in `token_locality_gate_v3.py` replaces the scalar gate with a K-dim vector per (token, head). The penalty becomes a dot product with precomputed RBF features of patch distances, allowing each token to learn arbitrary distance tuning curves. To enable it in training:

1. In the gate module constructor, replace `LocalityGateBranchV3` with `VectorKernelGateBranch`
2. Swap `_make_gated_attention_forward_v3` for `_make_vector_kernel_forward`
3. Add `reg_type: token_locality_v3_vec` to configs
4. Wire the new type into `train_cub.py`'s gate installation block (same pattern as v3)

### Add a new dataset

The training loop in `train_cub.py` expects a `CUBDataset`-style wrapper. Any HuggingFace image classification dataset with an `image` column and a `label` column will work by just changing `hf_dataset` in the config. For non-HuggingFace data, implement `__len__` and `__getitem__` matching the `CUBDataset` interface.
