"""Generate all visualisations from evaluation results.

Usage:
  python visualize.py \
      --baseline_results runs/baseline/eval/evaluation_results.json \
      --targeted_results runs/targeted/eval/evaluation_results.json \
      --baseline_log runs/baseline/training_log.json \
      --targeted_log runs/targeted/training_log.json \
      --output_dir visualizations/
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from common.plot_utils import (
    plot_training_curves,
    plot_mad_heatmap,
    plot_comparison_heatmaps,
    plot_mad_distributions,
    plot_mad_trajectories,
    plot_mad_variance,
    plot_local_mass_heatmaps,
    plot_entropy_heatmap,
    plot_distance_histograms,
    plot_bimodality_histogram,
    save_summary_table,
)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def extract_mad_dict(attn_stats):
    """Convert {str(layer): {mad: [...]}} to {int(layer): np.array}."""
    return {int(k): np.array(v["mad"]) for k, v in attn_stats.items()}


def extract_metric_dict(attn_stats, key):
    return {int(k): np.array(v[key]) for k, v in attn_stats.items() if key in v}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_results", type=str, required=True)
    parser.add_argument("--targeted_results", type=str, default=None)
    parser.add_argument("--baseline_log", type=str, required=True)
    parser.add_argument("--targeted_log", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="visualizations")
    args = parser.parse_args()

    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    base_res = load_json(args.baseline_results)
    base_log = load_json(args.baseline_log)
    base_attn = base_res["attention_stats"]
    base_mad = extract_mad_dict(base_attn)

    has_targeted = args.targeted_results is not None
    if has_targeted:
        tgt_res = load_json(args.targeted_results)
        tgt_log = load_json(args.targeted_log)
        tgt_attn = tgt_res["attention_stats"]
        tgt_mad = extract_mad_dict(tgt_attn)

    # 1. Training curves
    plot_training_curves(base_log, os.path.join(out, "baseline_curves.png"),
                         title_prefix="Baseline ")
    if has_targeted:
        plot_training_curves(tgt_log, os.path.join(out, "targeted_curves.png"),
                             title_prefix="Targeted ")

    # 2. MAD heatmaps
    plot_mad_heatmap(base_mad, os.path.join(out, "mad_heatmap_baseline.png"),
                     title="Baseline MAD")
    if has_targeted:
        plot_mad_heatmap(tgt_mad, os.path.join(out, "mad_heatmap_targeted.png"),
                         title="Targeted MAD")
        plot_comparison_heatmaps(base_mad, tgt_mad,
                                 os.path.join(out, "mad_comparison.png"))

    # 3. MAD distribution by layer
    if has_targeted:
        plot_mad_distributions(base_mad, tgt_mad,
                               os.path.join(out, "mad_distributions.png"))

    # 4. MAD trajectory over training
    base_epoch_mads = [e.get("mads", {}) for e in base_log]
    base_epoch_mads = [{int(k): v for k, v in m.items()} for m in base_epoch_mads if m]
    if base_epoch_mads:
        plot_mad_trajectories(base_epoch_mads,
                              os.path.join(out, "mad_trajectory_baseline.png"),
                              title_prefix="Baseline ")
    if has_targeted:
        tgt_epoch_mads = [e.get("mads", {}) for e in tgt_log]
        tgt_epoch_mads = [{int(k): v for k, v in m.items()} for m in tgt_epoch_mads if m]
        if tgt_epoch_mads:
            plot_mad_trajectories(tgt_epoch_mads,
                                  os.path.join(out, "mad_trajectory_targeted.png"),
                                  title_prefix="Targeted ")

    # 5. Inter-head MAD variance
    if has_targeted:
        plot_mad_variance(base_mad, tgt_mad,
                          os.path.join(out, "mad_variance.png"))

    # 6. Local mass heatmaps
    for tau in [0.15, 0.25, 0.35]:
        key = f"lm_{tau}"
        base_lm = extract_metric_dict(base_attn, key)
        if not base_lm:
            continue
        if has_targeted:
            tgt_lm = extract_metric_dict(tgt_attn, key)
            plot_local_mass_heatmaps(base_lm, tgt_lm, tau,
                                     os.path.join(out, f"local_mass_tau{tau}.png"))
        else:
            plot_mad_heatmap(base_lm,
                             os.path.join(out, f"local_mass_tau{tau}_baseline.png"),
                             title=f"Baseline Local Mass (τ={tau})")

    # 7. Entropy heatmap
    base_ent = extract_metric_dict(base_attn, "entropy")
    if has_targeted:
        tgt_ent = extract_metric_dict(tgt_attn, "entropy")
        plot_entropy_heatmap(base_ent, tgt_ent,
                             os.path.join(out, "entropy_comparison.png"))
    else:
        plot_mad_heatmap(base_ent, os.path.join(out, "entropy_baseline.png"),
                         title="Baseline Attention Entropy")

    # 8. Distance histograms (first 3 layers)
    if has_targeted:
        for b in list(base_attn.keys())[:3]:
            bi = int(b)
            bh = np.array(base_attn[b].get("dist_hist", []))
            th = np.array(tgt_attn[b].get("dist_hist", []))
            if bh.size > 0 and th.size > 0:
                plot_distance_histograms(bh, th, bi,
                                         os.path.join(out, f"dist_hist_layer{bi}.png"))

    # 11. Bimodality diagnostics
    all_base_mads = []
    for b in sorted(base_mad.keys()):
        all_base_mads.extend(base_mad[b].tolist())
    plot_bimodality_histogram(np.array(all_base_mads),
                              os.path.join(out, "bimodality_baseline.png"),
                              title="Baseline MAD Distribution")
    if has_targeted:
        all_tgt_mads = []
        for b in sorted(tgt_mad.keys()):
            all_tgt_mads.extend(tgt_mad[b].tolist())
        plot_bimodality_histogram(np.array(all_tgt_mads),
                                  os.path.join(out, "bimodality_targeted.png"),
                                  title="Targeted MAD Distribution")

    # 12. Summary table
    if has_targeted:
        save_summary_table(
            base_res.get("summary", {}),
            tgt_res.get("summary", {}),
            out,
        )

    print(f"Visualisations saved to {out}/")


if __name__ == "__main__":
    main()
