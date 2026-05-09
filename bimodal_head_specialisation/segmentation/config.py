"""Segmentation-specific configuration (ADE20K + DeiT-S encoder)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import config as base_cfg

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = base_cfg.PROJECT_ROOT
DATA_ROOT = "/sudarshana/data/ADEChallengeData2016"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "seg_outputs")

BASELINE_SEG_DIR = os.path.join(OUTPUT_DIR, "baseline_seg")
REGULARIZED_SEG_DIR = os.path.join(OUTPUT_DIR, "regularized_seg")
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "analysis")
REPORT_DIR = os.path.join(OUTPUT_DIR, "report")

# ─── Device ──────────────────────────────────────────────────────────────────
DEVICE = base_cfg.DEVICE  # "cuda:0", use CUDA_VISIBLE_DEVICES=2 externally

# ─── Model ───────────────────────────────────────────────────────────────────
MODEL_NAME = "deit_small_patch16_224"  # timm name — we override img_size at creation
NUM_SEG_CLASSES = 150     # ADE20K: 150 semantic classes
IGNORE_INDEX = 255        # unlabeled pixels
EMBED_DIM = 384
NUM_HEADS = 6
NUM_BLOCKS = 12
PATCH_SIZE = 16
IMG_SIZE = 512            # ADE20K standard training resolution
GRID_H = IMG_SIZE // PATCH_SIZE  # 32
GRID_W = IMG_SIZE // PATCH_SIZE  # 32
NUM_PATCHES = GRID_H * GRID_W   # 1024

# ─── Regularizer (STRONGER than v1) ─────────────────────────────────────────
REGULARIZED_BLOCKS = list(range(0, 6))  # blocks 0-5
NUM_LOW_HEADS = 3
NUM_HIGH_HEADS = 3
DELTA = 0.3
LAMBDA_GAP = 1.0          # 10× stronger than v1
LAMBDA_COMPACT = 0.1      # 10× stronger than v1
WARMUP_EPOCHS = 10         # longer warmup

# ─── Training ────────────────────────────────────────────────────────────────
EPOCHS = 40
BATCH_SIZE = 8             # 512×512 → 1024 tokens, attention caching is memory-heavy
BACKBONE_LR = 1e-5
DECODER_LR = 1e-4
WEIGHT_DECAY = 0.01
NUM_WORKERS = 0

# ─── Augmentation ────────────────────────────────────────────────────────────
SCALE_RANGE = (0.5, 2.0)
CROP_SIZE = 512

# ─── Analysis ────────────────────────────────────────────────────────────────
NUM_EVAL_BATCHES = 50
LOCAL_RADIUS_TAU = 5        # larger grid (32×32), so larger tau in patch units
NUM_MASK_TRIALS = 5
BOUNDARY_THICKNESS = 2      # pixels for boundary mask extraction

# ─── Normalization ───────────────────────────────────────────────────────────
IMAGENET_MEAN = base_cfg.IMAGENET_MEAN
IMAGENET_STD = base_cfg.IMAGENET_STD
