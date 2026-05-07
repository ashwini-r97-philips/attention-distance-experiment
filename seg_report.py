"""Generate segmentation experiment report with all figures and tables."""

import argparse
import json
import os

import numpy as np

import seg_config as cfg
from plot_utils import (
    plot_comparison_heatmaps,
    plot_seg_training_curves,
    plot_seg_head_masking,
    plot_distance_histograms,
    plot_conditional_mad,
)


def generate_report(eval_results_path, baseline_log_path, regularized_log_path, output_dir=None):
    output_dir = output_dir or cfg.REPORT_DIR
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    with open(eval_results_path) as f:
        results = json.load(f)

    baseline = results["baseline"]
    regularized = results["regularized"]
    all_blocks = list(range(cfg.NUM_BLOCKS))

    # 1. MAD comparison heatmaps
    base_mad = {b: np.array(baseline["metrics"][str(b)]["mad"]) for b in all_blocks}
    reg_mad = {b: np.array(regularized["metrics"][str(b)]["mad"]) for b in all_blocks}
    plot_comparison_heatmaps(base_mad, reg_mad, os.path.join(output_dir, "figures", "mad_comparison.png"))

    # 2. Local mass comparison
    base_lm = {b: np.array(baseline["metrics"][str(b)]["local_mass"]) for b in all_blocks}
    reg_lm = {b: np.array(regularized["metrics"][str(b)]["local_mass"]) for b in all_blocks}
    plot_comparison_heatmaps(base_lm, reg_lm, os.path.join(output_dir, "figures", "local_mass_comparison.png"))

    # 3. Entropy comparison
    base_ent = {b: np.array(baseline["metrics"][str(b)]["entropy"]) for b in all_blocks}
    reg_ent = {b: np.array(regularized["metrics"][str(b)]["entropy"]) for b in all_blocks}
    plot_comparison_heatmaps(base_ent, reg_ent, os.path.join(output_dir, "figures", "entropy_comparison.png"))

    # 4. Training curves
    if baseline_log_path and os.path.exists(baseline_log_path):
        plot_seg_training_curves(baseline_log_path, os.path.join(output_dir, "figures", "baseline_curves.png"))
    if regularized_log_path and os.path.exists(regularized_log_path):
        plot_seg_training_curves(regularized_log_path, os.path.join(output_dir, "figures", "regularized_curves.png"))

    # 5. Head masking
    plot_seg_head_masking(
        baseline["masking"], regularized["masking"],
        os.path.join(output_dir, "figures", "head_masking.png")
    )

    # 6. Distance histograms
    for bidx in cfg.REGULARIZED_BLOCKS[:3]:  # first 3 blocks
        bh = np.array(baseline["metrics"][str(bidx)]["dist_hist"])
        rh = np.array(regularized["metrics"][str(bidx)]["dist_hist"])
        plot_distance_histograms(
            bh, rh, bidx,
            os.path.join(output_dir, "figures", f"dist_hist_block{bidx}.png")
        )

    # 7. Conditional MAD
    plot_conditional_mad(
        baseline["conditional_mad"], regularized["conditional_mad"],
        os.path.join(output_dir, "figures", "conditional_mad.png")
    )

    # ─── Markdown Report ─────────────────────────────────────────────────────
    L = []
    L.append("# Bimodal Head Specialization — ADE20K Segmentation Report\n")
    L.append(f"Model: DeiT-S + Linear Decoder | Input: {cfg.IMG_SIZE}×{cfg.IMG_SIZE} | "
             f"Grid: {cfg.GRID_H}×{cfg.GRID_W} = {cfg.NUM_PATCHES} tokens\n")
    L.append(f"Regularized blocks: {cfg.REGULARIZED_BLOCKS} | "
             f"λ_gap={cfg.LAMBDA_GAP} | λ_compact={cfg.LAMBDA_COMPACT} | "
             f"δ={cfg.DELTA} | warmup={cfg.WARMUP_EPOCHS} epochs\n")

    # Segmentation accuracy
    L.append("\n## 1. Segmentation Performance\n")
    L.append("| Model | mIoU | Boundary F1 | mIoU (mask local) | bF1 (mask local) | mIoU (mask global) | bF1 (mask global) | mIoU (mask random) | bF1 (mask random) |")
    L.append("|-------|-----:|----------:|------------------:|---------------:|------------------:|----------------:|------------------:|----------------:|")
    for name, res in [("Baseline", baseline), ("Regularized", regularized)]:
        m = res["masking"]
        L.append(f"| {name} | {m['no_mask']['miou']:.4f} | {m['no_mask']['boundary_f1']:.4f} | "
                 f"{m['mask_local']['miou']:.4f} | {m['mask_local']['boundary_f1']:.4f} | "
                 f"{m['mask_global']['miou']:.4f} | {m['mask_global']['boundary_f1']:.4f} | "
                 f"{m['mask_random']['miou']:.4f} | {m['mask_random']['boundary_f1']:.4f} |")

    # Head masking differential
    L.append("\n### Head Masking Impact (drop from no-mask baseline)\n")
    L.append("| Model | Δ mIoU (local) | Δ bF1 (local) | Δ mIoU (global) | Δ bF1 (global) |")
    L.append("|-------|---------------:|--------------:|----------------:|---------------:|")
    for name, res in [("Baseline", baseline), ("Regularized", regularized)]:
        m = res["masking"]
        L.append(f"| {name} | "
                 f"{m['mask_local']['miou'] - m['no_mask']['miou']:+.4f} | "
                 f"{m['mask_local']['boundary_f1'] - m['no_mask']['boundary_f1']:+.4f} | "
                 f"{m['mask_global']['miou'] - m['no_mask']['miou']:+.4f} | "
                 f"{m['mask_global']['boundary_f1'] - m['no_mask']['boundary_f1']:+.4f} |")

    # GMM
    L.append("\n## 2. GMM Bimodality Test\n")
    L.append("| Block | Baseline BIC diff | Regularized BIC diff | Winner |")
    L.append("|-------|------------------:|--------------------:|--------|")
    for bidx in cfg.REGULARIZED_BLOCKS:
        b_bic = baseline["gmm"][str(bidx)]["bic_diff"]
        r_bic = regularized["gmm"][str(bidx)]["bic_diff"]
        winner = "Regularized" if r_bic > b_bic else "Baseline"
        L.append(f"| Block {bidx} | {b_bic:+.1f} | {r_bic:+.1f} | {winner} |")

    # Persistence
    L.append("\n## 3. Role Persistence\n")
    L.append("| Block | Baseline | Regularized |")
    L.append("|-------|--------:|----------:|")
    for bidx in cfg.REGULARIZED_BLOCKS:
        bp = baseline["persistence"][str(bidx)]["mean_persistence"]
        rp = regularized["persistence"][str(bidx)]["mean_persistence"]
        L.append(f"| Block {bidx} | {bp:.3f} | {rp:.3f} |")

    # Conditional MAD
    L.append("\n## 4. Boundary vs Interior MAD\n")
    L.append("Average MAD for query tokens at boundaries vs interior:\n")
    for bidx in cfg.REGULARIZED_BLOCKS:
        bc = baseline["conditional_mad"][str(bidx)]
        rc = regularized["conditional_mad"][str(bidx)]
        L.append(f"**Block {bidx}**:")
        L.append(f"  - Baseline:    boundary=[{', '.join(f'{v:.4f}' for v in bc['boundary_mad'])}]  "
                 f"interior=[{', '.join(f'{v:.4f}' for v in bc['interior_mad'])}]")
        L.append(f"  - Regularized: boundary=[{', '.join(f'{v:.4f}' for v in rc['boundary_mad'])}]  "
                 f"interior=[{', '.join(f'{v:.4f}' for v in rc['interior_mad'])}]\n")

    # MAD summary
    L.append("\n## 5. MAD Summary (Early Blocks)\n")
    for bidx in cfg.REGULARIZED_BLOCKS:
        bm = base_mad[bidx]
        rm = reg_mad[bidx]
        L.append(f"**Block {bidx}**: Baseline range={bm.max()-bm.min():.4f}, Regularized range={rm.max()-rm.min():.4f}")
        L.append(f"  - Baseline:    [{', '.join(f'{v:.4f}' for v in bm)}]")
        L.append(f"  - Regularized: [{', '.join(f'{v:.4f}' for v in rm)}]\n")

    # Head norms
    L.append("\n## 6. Head Output Norms\n")
    L.append("Verifying no head is dead (zero norm):\n")
    for bidx in cfg.REGULARIZED_BLOCKS:
        bn = baseline["head_norms"][str(bidx)]
        rn = regularized["head_norms"][str(bidx)]
        L.append(f"**Block {bidx}**: baseline=[{', '.join(f'{v:.3f}' for v in bn)}]  "
                 f"regularized=[{', '.join(f'{v:.3f}' for v in rn)}]")

    # Figures
    L.append("\n## 7. Figures\n")
    L.append("![MAD Comparison](figures/mad_comparison.png)\n")
    L.append("![Local Mass](figures/local_mass_comparison.png)\n")
    L.append("![Entropy](figures/entropy_comparison.png)\n")
    L.append("![Head Masking](figures/head_masking.png)\n")
    L.append("![Conditional MAD](figures/conditional_mad.png)\n")
    for bidx in cfg.REGULARIZED_BLOCKS[:3]:
        L.append(f"![Distance Histogram Block {bidx}](figures/dist_hist_block{bidx}.png)\n")
    L.append("![Baseline Curves](figures/baseline_curves.png)\n")
    L.append("![Regularized Curves](figures/regularized_curves.png)\n")

    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(L))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_results", type=str, required=True)
    parser.add_argument("--baseline_log", type=str, default=None)
    parser.add_argument("--regularized_log", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    bl = args.baseline_log or os.path.join(cfg.BASELINE_SEG_DIR, "training_log.json")
    rl = args.regularized_log or os.path.join(cfg.REGULARIZED_SEG_DIR, "training_log.json")
    generate_report(args.eval_results, bl, rl, args.output_dir)
