#!/usr/bin/env bash
# Run all post-training analyses on baseline and (optionally) targeted checkpoints.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_analyses.sh
#
# Override configs/checkpoints:
#   BASELINE_CONFIG=configs/baseline.yaml \
#   TARGETED_CONFIG=configs/spread_weak.yaml \
#   bash run_analyses.sh
#
# Run only specific analyses:
#   ANALYSES="cka,bimodality" bash run_analyses.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASELINE_CONFIG="${BASELINE_CONFIG:-configs/baseline.yaml}"
TARGETED_CONFIG="${TARGETED_CONFIG:-}"

BASELINE_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$BASELINE_CONFIG'))['output_dir'])")
BASELINE_CKPT="${BASELINE_DIR}/checkpoints/best.pth"

TARGETED_ARGS=""
if [[ -n "$TARGETED_CONFIG" ]]; then
    TARGETED_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$TARGETED_CONFIG'))['output_dir'])")
    TARGETED_CKPT="${TARGETED_DIR}/checkpoints/best.pth"
    TARGETED_ARGS="--targeted_ckpt ${TARGETED_CKPT}"
fi

# Which analyses to run (comma-separated, default: all)
ANALYSES="${ANALYSES:-cka,svcca,rollout,llc,bimodality}"

echo "=============================================="
echo "  Post-Training Analysis Suite"
echo "=============================================="
echo "Baseline config : $BASELINE_CONFIG"
echo "Baseline ckpt   : $BASELINE_CKPT"
if [[ -n "$TARGETED_CONFIG" ]]; then
    echo "Targeted config : $TARGETED_CONFIG"
    echo "Targeted ckpt   : $TARGETED_CKPT"
else
    echo "Targeted        : (none — baseline-only analyses)"
fi
echo "Analyses        : $ANALYSES"
echo "Device          : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "=============================================="

run_analysis() {
    local name="$1"
    local script="$2"
    shift 2
    if [[ ",$ANALYSES," == *",$name,"* ]]; then
        echo ""
        echo "────────────────────────────────────────"
        echo "  Running: $name"
        echo "────────────────────────────────────────"
        python "$script" "$@"
    else
        echo "  Skipping: $name"
    fi
}

# 1. CKA — representation similarity
run_analysis cka analysis_cka.py \
    --config "$BASELINE_CONFIG" \
    --baseline_ckpt "$BASELINE_CKPT" \
    $TARGETED_ARGS \
    --output_dir "${BASELINE_DIR}/analysis/cka"

# 2. SVCCA — subspace alignment
run_analysis svcca analysis_svcca.py \
    --config "$BASELINE_CONFIG" \
    --baseline_ckpt "$BASELINE_CKPT" \
    $TARGETED_ARGS \
    --output_dir "${BASELINE_DIR}/analysis/svcca"

# 3. Attention rollout / flow / GMAR
run_analysis rollout analysis_rollout.py \
    --config "$BASELINE_CONFIG" \
    --baseline_ckpt "$BASELINE_CKPT" \
    $TARGETED_ARGS \
    --output_dir "${BASELINE_DIR}/analysis/rollout"

# 4. LLC — per-head SLT complexity (most expensive)
run_analysis llc analysis_llc.py \
    --config "$BASELINE_CONFIG" \
    --checkpoint "$BASELINE_CKPT" \
    --output_dir "${BASELINE_DIR}/analysis/llc" \
    --sgld_steps 500 \
    --num_chains 3

if [[ -n "$TARGETED_CONFIG" ]]; then
    run_analysis llc analysis_llc.py \
        --config "$TARGETED_CONFIG" \
        --checkpoint "$TARGETED_CKPT" \
        --output_dir "${TARGETED_DIR}/analysis/llc" \
        --sgld_steps 500 \
        --num_chains 3
fi

# 5. MAD bimodality — histograms, GMM, dip test, stability
run_analysis bimodality analysis_mad_bimodality.py \
    --config "$BASELINE_CONFIG" \
    --baseline_ckpt "$BASELINE_CKPT" \
    $TARGETED_ARGS \
    --output_dir "${BASELINE_DIR}/analysis/bimodality"

echo ""
echo "=============================================="
echo "  All analyses complete!"
echo "  Results: ${BASELINE_DIR}/analysis/"
if [[ -n "$TARGETED_CONFIG" ]]; then
    echo "           ${TARGETED_DIR}/analysis/"
fi
echo "=============================================="
