"""YAML-based configuration for attention-distance experiments.

All experiment scripts load config via:
    cfg = load_config("path/to/config.yaml")
which returns a namespace object with attribute access.
"""

import os
import yaml
from types import SimpleNamespace


# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULTS = {
    # Paths
    "dataset_root": "/sudarshana/data/imagenet",
    "output_dir": None,  # set per-run

    # Device
    "device": "cuda:0",

    # Model
    "model_name": "vit_small_patch16_224",
    "num_classes": 1000,
    "embed_dim": 384,
    "num_heads": 6,
    "num_blocks": 12,
    "patch_size": 16,
    "img_size": 224,

    # Training
    "epochs": 90,
    "batch_size": 256,
    "lr": 1e-4,
    "weight_decay": 0.05,
    "warmup_epochs": 5,
    "label_smoothing": 0.1,
    "mixup_alpha": 0.8,
    "cutmix_alpha": 1.0,
    "mixup_prob": 1.0,
    "mixup_switch_prob": 0.5,
    "num_workers": 8,
    "seed": 42,
    "grad_clip_norm": 1.0,

    # Regulariser
    "reg_type": "none",          # none | spread | bimodal
    "lambda_reg": 0.0,
    "lambda_warmup_epochs": 5,
    "regularized_blocks": list(range(12)),

    # Bimodal-loss specific
    "m_loc": 0.20,
    "m_glob": 0.65,
    "sigma_loc": 0.08,
    "sigma_glob": 0.12,
    "pi_mix": 0.5,

    # Attention analysis
    "tau_values": [0.15, 0.25, 0.35],
    "attention_eval_subset_size": 1024,
    "attention_eval_frequency_epochs": 1,
    "attention_map_frequency_epochs": 5,
    "num_vis_images": 16,
    "num_vis_query_patches": 4,

    # ImageNet normalisation
    "imagenet_mean": [0.485, 0.456, 0.406],
    "imagenet_std": [0.229, 0.224, 0.225],
}


def _add_derived(cfg):
    """Add derived constants that depend on other config values."""
    cfg.grid_h = cfg.img_size // cfg.patch_size
    cfg.grid_w = cfg.img_size // cfg.patch_size
    cfg.num_patches = cfg.grid_h * cfg.grid_w
    if not hasattr(cfg, "project_root"):
        cfg.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return cfg


def load_config(yaml_path=None, overrides=None):
    """Load config from YAML, falling back to defaults."""
    d = dict(DEFAULTS)
    if yaml_path and os.path.exists(yaml_path):
        with open(yaml_path) as f:
            user = yaml.safe_load(f) or {}
        d.update(user)
    if overrides:
        d.update(overrides)
    cfg = SimpleNamespace(**d)
    _add_derived(cfg)
    return cfg


def save_config(cfg, path):
    """Save config to YAML."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    for k, v in d.items():
        if isinstance(v, tuple):
            d[k] = list(v)
    with open(path, "w") as f:
        yaml.dump(d, f, default_flow_style=False, sort_keys=False)
