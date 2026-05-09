"""ImageNet-1K data loading with standard ViT augmentation.

Expected folder structure:
    dataset_root/
        train/
            n01440764/
            n01443537/
            ...
        val/
            n01440764/
            ...
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def build_train_transform(cfg):
    return transforms.Compose([
        transforms.RandomResizedCrop(cfg.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])


def build_val_transform(cfg):
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.imagenet_mean, std=cfg.imagenet_std),
    ])


def get_train_dataset(cfg):
    train_dir = os.path.join(cfg.dataset_root, "train")
    return datasets.ImageFolder(train_dir, transform=build_train_transform(cfg))


def get_val_dataset(cfg):
    val_dir = os.path.join(cfg.dataset_root, "val")
    return datasets.ImageFolder(val_dir, transform=build_val_transform(cfg))


def get_train_loader(cfg):
    ds = get_train_dataset(cfg)
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        generator=g,
        persistent_workers=cfg.num_workers > 0,
    )


def get_val_loader(cfg, batch_size=None):
    ds = get_val_dataset(cfg)
    return DataLoader(
        ds,
        batch_size=batch_size or cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=cfg.num_workers > 0,
    )


def get_attention_eval_subset(cfg):
    """Return a fixed-index subset of the validation set for attention analysis."""
    val_ds = get_val_dataset(cfg)
    n = min(cfg.attention_eval_subset_size, len(val_ds))
    rng = np.random.RandomState(cfg.seed)
    indices = rng.choice(len(val_ds), size=n, replace=False)
    indices.sort()
    sub = Subset(val_ds, indices.tolist())
    loader = DataLoader(
        sub,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return loader, indices.tolist()
