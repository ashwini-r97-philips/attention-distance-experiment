"""Generate final comparison report with all figures and markdown summary."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common import config as cfg
from common.plot_utils import plot_comparison_heatmaps, plot_training_curves, plot_head_masking_results


def generate_report(eval_results_path, baseline_log_path, regularized_log_path, output_dir=None):
    output_dir = output_dir or cfg.REPORT_DIR
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    with open(eval_results_path) as f:
        results = json.load(f)

    baseline = results["baseline"]
    regularized = results["regularized"]

    # 1. Comparison MAD heatmaps
    all_blocks = list(range(cfg.NUM_BLOCKS))
    base_mad = {b: np.array(baseline["metrics"][str(b)]["mad"]) for b in all_blocks}
    reg_mad = {b: np.array(regularized["metrics"][str(b)]["mad"]) for b in all_blocks}

    plot_comparison_heatmaps(
        base_mad, reg_mad,
        os.path.join(output_dir, "figures", "mad_comparison.png")
    )

    # 2. Training curves
    if baseline_log_path and os.path.exists(baseline_log_path):
        plot_training_curves(baseline_log_path, os.path.join(output_dir, "figures", "baseline_curves.png"))
    if regularized_log_path and os.path.exists(regularized_log_path):
        plot_training_curves(regularized_log_path, os.path.join(output_dir, "figures", "regularized_curves.png"))

    # 3. Head masking results
    mask_results = {
        "baseline": baseline["masking"],
        "regularized": regularized["masking"],
    }
    plot_head_masking_results(mask_results, os.path.join(output_dir, "figures", "head_masking.png"))

    # 4. Markdown report
    report_lines = []
    report_lines.append("# Bimodal Head Specialization Experiment Report\n")
    report_lines.append(f"Model: {cfg.MODEL_NAME} | Regularized blocks: {cfg.REGULARIZED_BLOCKS}\n")
    report_lines.append(f"Delta: {cfg.DELTA} | λ_gap: {cfg.LAMBDA_GAP} | λ_compact: {cfg.LAMBDA_COMPACT}\n")

    # Accuracy
    report_lines.append("\n## 1. Accuracy\n")
    report_lines.append("| Model | Val Top-1 (no mask) | Mask Local | Mask Global | Mask Random |")
    report_lines.append("|-------|--------------------:|----------:|----------:|----------:|")
    for name, res in [("Baseline FT", baseline), ("Regularized FT", regularized)]:
        m = res["masking"]
        report_lines.append(f"| {name} | {m['no_mask']:.2f}% | {m['mask_local']:.2f}% | "
                          f"{m['mask_global']:.2f}% | {m['mask_random']:.2f}% |")

    # GMM Bimodality
    report_lines.append("\n## 2. GMM Bimodality Test\n")
    report_lines.append("BIC difference (positive = 2-component model preferred):\n")
    report_lines.append("| Block | Baseline BIC diff | Regularized BIC diff | Winner |")
    report_lines.append("|-------|------------------:|--------------------:|--------|")
    for bidx in cfg.REGULARIZED_BLOCKS:
        b_bic = baseline["gmm"][str(bidx)]["bic_diff"]
        r_bic = regularized["gmm"][str(bidx)]["bic_diff"]
        winner = "Regularized" if r_bic > b_bic else "Baseline"
        report_lines.append(f"| Block {bidx} | {b_bic:+.1f} | {r_bic:+.1f} | {winner} |")

    # Role Persistence
    report_lines.append("\n## 3. Role Persistence\n")
    report_lines.append("Mean persistence (fraction of batches where head keeps its dominant role):\n")
    report_lines.append("| Block | Baseline | Regularized |")
    report_lines.append("|-------|--------:|----------:|")
    for bidx in cfg.REGULARIZED_BLOCKS:
        b_p = baseline["persistence"][str(bidx)]["mean_persistence"]
        r_p = regularized["persistence"][str(bidx)]["mean_persistence"]
        report_lines.append(f"| Block {bidx} | {b_p:.3f} | {r_p:.3f} |")

    # MAD summary
    report_lines.append("\n## 4. MAD Summary (Early Blocks)\n")
    report_lines.append("Per-head MAD values:\n")
    for bidx in cfg.REGULARIZED_BLOCKS:
        b_m = base_mad[bidx]
        r_m = reg_mad[bidx]
        b_range = b_m.max() - b_m.min()
        r_range = r_m.max() - r_m.min()
        report_lines.append(f"**Block {bidx}**: Baseline range={b_range:.4f}, Regularized range={r_range:.4f}")
        report_lines.append(f"  - Baseline:    [{', '.join(f'{v:.4f}' for v in b_m)}]")
        report_lines.append(f"  - Regularized: [{', '.join(f'{v:.4f}' for v in r_m)}]\n")

    # Figures
    report_lines.append("\n## 5. Figures\n")
    report_lines.append("![MAD Comparison](figures/mad_comparison.png)\n")
    report_lines.append("![Head Masking](figures/head_masking.png)\n")
    if baseline_log_path and os.path.exists(baseline_log_path):
        report_lines.append("![Baseline Curves](figures/baseline_curves.png)\n")
    if regularized_log_path and os.path.exists(regularized_log_path):
        report_lines.append("![Regularized Curves](figures/regularized_curves.png)\n")

    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Report saved to {report_path}")
    print(f"Figures saved to {os.path.join(output_dir, 'figures')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_results", type=str, required=True)
    parser.add_argument("--baseline_log", type=str, default=None)
    parser.add_argument("--regularized_log", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    baseline_log = args.baseline_log or os.path.join(cfg.BASELINE_FT_DIR, "training_log.json")
    reg_log = args.regularized_log or os.path.join(cfg.REGULARIZED_FT_DIR, "training_log.json")

    generate_report(args.eval_results, baseline_log, reg_log, args.output_dir)
