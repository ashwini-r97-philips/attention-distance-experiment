#!/bin/bash
# Master script for ADE20K segmentation bimodal head specialization experiment.
# Usage: CUDA_VISIBLE_DEVICES=2 bash run_seg_experiment.sh
set -euo pipefail

cd "$(dirname "$0")"

CONDA_BASE="${HOME}/.conda/envs/jepa-medsam"
export PATH="${CONDA_BASE}/bin:${PATH}"

echo "============================================================"
echo "Phase 0: Download ADE20K (if not present)"
echo "============================================================"
if [ ! -d "/sudarshana/data/ADEChallengeData2016" ]; then
    bash download_ade20k.sh /sudarshana/data/
fi
echo "ADE20K ready at /sudarshana/data/ADEChallengeData2016"

echo ""
echo "============================================================"
echo "Phase 1: Baseline segmentation training (30 epochs)"
echo "============================================================"
python seg_train.py --mode baseline --epochs 30

echo ""
echo "============================================================"
echo "Phase 2: Regularized segmentation training (30 epochs)"
echo "============================================================"
python seg_train.py --mode regularized --epochs 30

echo ""
echo "============================================================"
echo "Phase 3: Evaluation"
echo "============================================================"
python seg_evaluate.py \
    --baseline_ckpt seg_outputs/baseline_seg/checkpoints/best.pth \
    --regularized_ckpt seg_outputs/regularized_seg/checkpoints/best.pth

echo ""
echo "============================================================"
echo "Phase 4: Report generation"
echo "============================================================"
python seg_report.py \
    --eval_results seg_outputs/analysis/seg_evaluation_results.json

echo ""
echo "============================================================"
echo "ALL DONE — Report at seg_outputs/report/report.md"
echo "============================================================"
