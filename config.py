"""Configuration for the bimodal head specialization experiment."""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = "/sudarshana/data/tiny-imagenet-200"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

BASELINE_ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "baseline_pretrained")
BASELINE_FT_DIR = os.path.join(OUTPUT_DIR, "baseline_ft")
REGULARIZED_FT_DIR = os.path.join(OUTPUT_DIR, "regularized_ft")
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "analysis")
REPORT_DIR = os.path.join(OUTPUT_DIR, "report")

# ─── Device ──────────────────────────────────────────────────────────────────
DEVICE = "cuda:0"  # We'll set CUDA_VISIBLE_DEVICES=2 externally

# ─── Model ───────────────────────────────────────────────────────────────────
MODEL_NAME = "deit_small_patch16_224"
NUM_CLASSES = 200
EMBED_DIM = 384
NUM_HEADS = 6
NUM_BLOCKS = 12
PATCH_SIZE = 16
IMG_SIZE = 224
GRID_H = IMG_SIZE // PATCH_SIZE  # 14
GRID_W = IMG_SIZE // PATCH_SIZE  # 14
NUM_PATCHES = GRID_H * GRID_W   # 196

# ─── Regularizer ─────────────────────────────────────────────────────────────
REGULARIZED_BLOCKS = list(range(0, 6))  # blocks 0-5
NUM_LOW_HEADS = 3   # bottom half
NUM_HIGH_HEADS = 3  # top half
DELTA = 0.3         # target gap between local and global groups
LAMBDA_GAP = 0.1
LAMBDA_COMPACT = 0.01
WARMUP_EPOCHS = 5   # linear ramp for regularizer weight

# ─── Training ────────────────────────────────────────────────────────────────
EPOCHS = 30
BATCH_SIZE = 256
LR = 5e-5
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.8
CUTMIX_ALPHA = 1.0
MIXUP_PROB = 1.0
MIXUP_SWITCH_PROB = 0.5
NUM_WORKERS = 8

# ─── Analysis ────────────────────────────────────────────────────────────────
NUM_EVAL_BATCHES = 50       # batches for analysis passes
LOCAL_RADIUS_TAU = 3        # in patch units for local mass
NUM_MASK_TRIALS = 5         # random head masking repeats
PERSISTENCE_BATCHES = 50    # batches for role persistence check

# ─── ImageNet normalization ──────────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
