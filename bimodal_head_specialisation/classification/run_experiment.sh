#!/usr/bin/env bash
# Run baseline + targeted experiments and generate visualisations.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_experiment.sh
#   CUDA_VISIBLE_DEVICES=0 bash run_experiment.sh --config configs/baseline.yaml
#
# Phases:
#   1. Baseline training
#   2. Targeted training (spread or bimodal)
#   3. Evaluation of both
#   4. Visualisation generation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASELINE_CONFIG="${BASELINE_CONFIG:-configs/baseline.yaml}"
TARGETED_CONFIG="${TARGETED_CONFIG:-configs/spread_weak.yaml}"

echo "=============================================="
echo "ViT-S/16 ImageNet-1K Attention Distance Experiment"
echo "=============================================="
echo "Baseline config : $BASELINE_CONFIG"
echo "Targeted config : $TARGETED_CONFIG"
echo "Device          : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "=============================================="

# Phase 1: Baseline
echo ""
echo "[Phase 1] Baseline training..."
python train.py --config "$BASELINE_CONFIG"

# Phase 2: Targeted
echo ""
echo "[Phase 2] Targeted training..."
python train.py --config "$TARGETED_CONFIG"

# Phase 3: Evaluate baseline
echo ""
echo "[Phase 3a] Evaluating baseline..."
BASELINE_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$BASELINE_CONFIG'))['output_dir'])")
python evaluate.py \
    --config "$BASELINE_CONFIG" \
    --checkpoint "$BASELINE_DIR/checkpoints/best.pth"

# Evaluate targeted
echo ""
echo "[Phase 3b] Evaluating targeted..."
TARGETED_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$TARGETED_CONFIG'))['output_dir'])")
python evaluate.py \
    --config "$TARGETED_CONFIG" \
    --checkpoint "$TARGETED_DIR/checkpoints/best.pth"

# Phase 4: Visualisations
echo ""
echo "[Phase 4] Generating visualisations..."
python visualize.py \
    --baseline_results "$BASELINE_DIR/eval/evaluation_results.json" \
    --targeted_results "$TARGETED_DIR/eval/evaluation_results.json" \
    --baseline_log "$BASELINE_DIR/training_log.json" \
    --targeted_log "$TARGETED_DIR/training_log.json" \
    --output_dir visualizations/

echo ""
echo "=============================================="
echo "Experiment complete!"
echo "  Baseline:  $BASELINE_DIR/"
echo "  Targeted:  $TARGETED_DIR/"
echo "  Plots:     visualizations/"
echo "=============================================="
