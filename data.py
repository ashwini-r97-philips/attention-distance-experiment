"""ImageNet / Tiny-ImageNet data loading with standard DeiT augmentation.

Supports:
  - Tiny-ImageNet-200 (auto-detected by val/val_annotations.txt)
  - Standard ImageFolder layout
  - HuggingFace datasets backend
"""

import os

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image

import config as cfg


def build_train_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomResizedCrop(cfg.IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD),
    ])


def build_val_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(cfg.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD),
    ])


# ─── Tiny-ImageNet support ───────────────────────────────────────────────────

class TinyImageNetValDataset(Dataset):
    """Tiny-ImageNet val set: flat images/ dir + val_annotations.txt mapping."""

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []

        # Build class-to-index mapping from train dir
        train_dir = os.path.join(os.path.dirname(root), "train")
        class_names = sorted(os.listdir(train_dir))
        self.class_to_idx = {cn: i for i, cn in enumerate(class_names)}

        # Parse val_annotations.txt
        ann_file = os.path.join(root, "val_annotations.txt")
        with open(ann_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                fname, class_name = parts[0], parts[1]
                img_path = os.path.join(root, "images", fname)
                self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def _is_tiny_imagenet(data_root):
    """Detect Tiny-ImageNet by presence of val/val_annotations.txt."""
    return os.path.exists(os.path.join(data_root, "val", "val_annotations.txt"))


def get_tiny_imagenet_train_dataset(data_root):
    """Tiny-ImageNet train: class/images/*.JPEG — use ImageFolder on class dirs."""
    train_dir = os.path.join(data_root, "train")

    # Tiny-ImageNet has train/classname/images/*.JPEG structure.
    # We need a custom dataset since ImageFolder expects train/classname/*.JPEG
    class TinyTrainDataset(Dataset):
        def __init__(self, root, transform=None):
            self.transform = transform
            self.samples = []
            class_names = sorted(os.listdir(root))
            self.class_to_idx = {cn: i for i, cn in enumerate(class_names)}
            for cn in class_names:
                img_dir = os.path.join(root, cn, "images")
                if not os.path.isdir(img_dir):
                    continue
                for fname in sorted(os.listdir(img_dir)):
                    if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                        self.samples.append((os.path.join(img_dir, fname), self.class_to_idx[cn]))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label

    return TinyTrainDataset(train_dir, transform=build_train_transform())


def get_tiny_imagenet_val_dataset(data_root):
    val_dir = os.path.join(data_root, "val")
    return TinyImageNetValDataset(val_dir, transform=build_val_transform())


# ─── HuggingFace datasets backend ───────────────────────────────────────────

class HFImageNetDataset(Dataset):
    """Wraps a HuggingFace Dataset split for use with PyTorch DataLoader."""

    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item["image"]
        label = item["label"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


_hf_cache = {}


def _load_hf_split(split):
    """Load and cache an HF dataset split."""
    if split not in _hf_cache:
        from datasets import load_dataset
        print(f"Loading ILSVRC/imagenet-1k split='{split}' from HuggingFace...")
        _hf_cache[split] = load_dataset(
            "ILSVRC/imagenet-1k",
            split=split,
            trust_remote_code=True,
        )
        print(f"  Loaded {len(_hf_cache[split])} samples")
    return _hf_cache[split]


def get_hf_train_dataset():
    return HFImageNetDataset(_load_hf_split("train"), transform=build_train_transform())


def get_hf_val_dataset():
    return HFImageNetDataset(_load_hf_split("validation"), transform=build_val_transform())


# ─── ImageFolder backend ────────────────────────────────────────────────────

def get_folder_train_dataset(data_root=None):
    root = data_root or cfg.DATA_ROOT
    train_dir = os.path.join(root, "train")
    return datasets.ImageFolder(train_dir, transform=build_train_transform())


def get_folder_val_dataset(data_root=None):
    root = data_root or cfg.DATA_ROOT
    val_dir = os.path.join(root, "val")
    return datasets.ImageFolder(val_dir, transform=build_val_transform())


# ─── Unified interface ──────────────────────────────────────────────────────

def _use_hf(data_root):
    """Decide whether to use HF datasets backend."""
    root = data_root or cfg.DATA_ROOT
    if _is_tiny_imagenet(root):
        return False
    if root and os.path.isdir(os.path.join(root, "train")):
        return False
    try:
        import datasets as hf_datasets  # noqa: F401
        return True
    except ImportError:
        return False


def get_train_dataset(data_root=None):
    root = data_root or cfg.DATA_ROOT
    if _is_tiny_imagenet(root):
        return get_tiny_imagenet_train_dataset(root)
    if _use_hf(data_root):
        return get_hf_train_dataset()
    return get_folder_train_dataset(data_root)


def get_val_dataset(data_root=None):
    root = data_root or cfg.DATA_ROOT
    if _is_tiny_imagenet(root):
        return get_tiny_imagenet_val_dataset(root)
    if _use_hf(data_root):
        return get_hf_val_dataset()
    return get_folder_val_dataset(data_root)


def get_train_loader(data_root=None, batch_size=None, num_workers=None):
    ds = get_train_dataset(data_root)
    return DataLoader(
        ds,
        batch_size=batch_size or cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers if num_workers is not None else cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )


def get_val_loader(data_root=None, batch_size=None, num_workers=None):
    ds = get_val_dataset(data_root)
    return DataLoader(
        ds,
        batch_size=batch_size or cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers if num_workers is not None else cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )


def get_debug_loaders(data_root=None, num_classes=10, batch_size=64):
    """Small subset (first num_classes) for debugging."""
    train_ds = get_train_dataset(data_root)
    val_ds = get_val_dataset(data_root)

    def _filter_indices(ds, num_classes):
        if isinstance(ds, HFImageNetDataset):
            return [i for i in range(len(ds)) if ds.hf_dataset[i]["label"] < num_classes]
        elif hasattr(ds, "samples"):
            return [i for i, (_, y) in enumerate(ds.samples) if y < num_classes]
        else:
            # Fallback: iterate (slow but works for any dataset)
            return [i for i in range(len(ds)) if ds[i][1] < num_classes]

    train_indices = _filter_indices(train_ds, num_classes)
    val_indices = _filter_indices(val_ds, num_classes)

    train_loader = DataLoader(
        Subset(train_ds, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader
