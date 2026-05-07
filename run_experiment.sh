#!/usr/bin/env bash
# Master script for the bimodal head specialization experiment.
# Usage: CUDA_VISIBLE_DEVICES=2 bash run_experiment.sh [--debug] [--data_root /path/to/imagenet]
#
# Phases:
#   1. Baseline analysis (pretrained, no training)
#   2. Baseline finetuning (30 epochs)
#   3. Regularized finetuning (30 epochs)
#   4. Post-training evaluation
#   5. Report generation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults
DATA_ROOT="${DATA_ROOT:-$HOME/datasets/imagenet-1k}"
DEBUG=""
EPOCHS=30
BATCH_SIZE=256

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug) DEBUG="--debug"; EPOCHS=3; BATCH_SIZE=64; shift ;;
        --data_root) DATA_ROOT="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "=============================================="
echo "Bimodal Head Specialization Experiment"
echo "=============================================="
echo "Data root:  $DATA_ROOT"
echo "Epochs:     $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Debug:      ${DEBUG:-no}"
echo "Device:     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "=============================================="

# Phase 1: Baseline analysis
echo ""
echo "[Phase 1] Baseline analysis of pretrained DeiT-S..."
python baseline_analysis.py \
    --data_root "$DATA_ROOT" \
    --batch_size 64 \
    --num_batches 50

# Phase 2: Baseline finetuning
echo ""
echo "[Phase 2] Baseline finetuning..."
python train.py \
    --mode baseline \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --data_root "$DATA_ROOT" \
    $DEBUG

# Phase 3: Regularized finetuning
echo ""
echo "[Phase 3] Regularized finetuning..."
python train.py \
    --mode regularized \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --data_root "$DATA_ROOT" \
    $DEBUG

# Phase 4: Evaluation
echo ""
echo "[Phase 4] Post-training evaluation..."
python evaluate.py \
    --baseline_ckpt outputs/baseline_ft/checkpoints/best.pth \
    --regularized_ckpt outputs/regularized_ft/checkpoints/best.pth \
    --data_root "$DATA_ROOT" \
    --batch_size 64

# Phase 5: Report
echo ""
echo "[Phase 5] Generating report..."
python generate_report.py \
    --eval_results outputs/analysis/evaluation_results.json

echo ""
echo "=============================================="
echo "Experiment complete!"
echo "Report: outputs/report/report.md"
echo "=============================================="
