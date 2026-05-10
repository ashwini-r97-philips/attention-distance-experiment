"""ImageNet-1K data loading via HuggingFace datasets (streaming).

Uses `datasets.load_dataset("ILSVRC/imagenet-1k", streaming=True)` so
ImageNet is never fully downloaded to disk.  The HF token must be set
via the HF_TOKEN env-var or `huggingface-cli login`.

For the attention-eval subset we materialise a small fixed slice of the
validation split into memory.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, Dataset, Subset
from torchvision import transforms
from datasets import load_dataset


# ─── Transforms ───────────────────────────────────────────────────────────────

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


# ─── HuggingFace streaming wrapper ───────────────────────────────────────────

class HFStreamingDataset(IterableDataset):
    """Wraps a HuggingFace IterableDataset for use with PyTorch DataLoader.

    Each example is expected to have an 'image' (PIL) and 'label' (int) field.
    Images are converted to RGB and transformed on the fly.
    """

    def __init__(self, hf_dataset, transform, seed=42, shuffle_buffer=10_000):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self):
        # Shuffle with a buffer for training; no-ops for val (buffer=0)
        ds = self.hf_dataset
        if self.shuffle_buffer > 0:
            ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Split the stream across DataLoader workers
            ds = _split_iterable_for_worker(ds, worker_info)

        for example in ds:
            img = example["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            label = example["label"]
            yield self.transform(img), label


def _split_iterable_for_worker(ds, worker_info):
    """Yield every n-th example so workers don't duplicate data."""
    worker_id = worker_info.id
    num_workers = worker_info.num_workers
    for i, item in enumerate(ds):
        if i % num_workers == worker_id:
            yield item


class HFMapDataset(Dataset):
    """Wraps a list of materialised HF examples as a map-style Dataset."""

    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img), ex["label"]


# ─── Dataset / Loader factories ───────────────────────────────────────────────

def _hf_name(cfg):
    return getattr(cfg, "hf_dataset", "ILSVRC/imagenet-1k")


def get_train_loader(cfg):
    ds = load_dataset(_hf_name(cfg), split="train", streaming=True,
                      trust_remote_code=True)
    wrapped = HFStreamingDataset(ds, build_train_transform(cfg),
                                  seed=cfg.seed, shuffle_buffer=10_000)
    return DataLoader(
        wrapped,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        # IterableDataset: shuffle is handled inside the dataset
    )


def get_val_loader(cfg, batch_size=None):
    ds = load_dataset(_hf_name(cfg), split="validation", streaming=True,
                      trust_remote_code=True)
    wrapped = HFStreamingDataset(ds, build_val_transform(cfg),
                                  seed=cfg.seed, shuffle_buffer=0)
    return DataLoader(
        wrapped,
        batch_size=batch_size or cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def get_attention_eval_subset(cfg):
    """Materialise a fixed-index subset of the validation set for attention analysis.

    Because the streaming dataset has no random access, we iterate through
    the validation split once, deterministically selecting
    `attention_eval_subset_size` examples spaced evenly across the 50 000
    validation images.
    """
    n_val = getattr(cfg, "n_val_images", 50_000)
    n = min(cfg.attention_eval_subset_size, n_val)
    rng = np.random.RandomState(cfg.seed)
    target_indices = set(sorted(rng.choice(n_val, size=n, replace=False)))

    ds = load_dataset(_hf_name(cfg), split="validation", streaming=True,
                      trust_remote_code=True)

    examples = []
    for i, ex in enumerate(ds):
        if i in target_indices:
            examples.append(ex)
        if len(examples) >= n:
            break

    subset = HFMapDataset(examples, build_val_transform(cfg))
    loader = DataLoader(
        subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=min(cfg.num_workers, 4),
        pin_memory=True,
    )
    return loader, sorted(target_indices)
