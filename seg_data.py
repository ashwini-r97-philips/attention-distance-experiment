"""ADE20K data loading with segmentation transforms."""

import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

import seg_config as cfg


class ADE20KDataset(Dataset):
    """ADE20K semantic segmentation dataset.

    Labels: 0 = unlabeled (mapped to IGNORE_INDEX), 1-150 = semantic classes.
    We shift labels to 0-149 (subtract 1), and map original 0 to IGNORE_INDEX.
    """

    def __init__(self, root, split="training", transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        img_dir = os.path.join(root, "images", split)
        ann_dir = os.path.join(root, "annotations", split)

        self.images = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.endswith(".jpg")
        ])
        self.annotations = sorted([
            os.path.join(ann_dir, f) for f in os.listdir(ann_dir)
            if f.endswith(".png")
        ])
        assert len(self.images) == len(self.annotations), \
            f"Mismatch: {len(self.images)} images vs {len(self.annotations)} annotations"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.annotations[idx])  # uint8/uint16, values 0-150

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask


class SegTrainTransform:
    """Training transform: random resize, random crop, horizontal flip, normalize."""

    def __init__(self, crop_size=None, scale_range=None):
        self.crop_size = crop_size or cfg.CROP_SIZE
        self.scale_range = scale_range or cfg.SCALE_RANGE
        self.normalize = transforms.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD)

    def __call__(self, image, mask):
        # Random scale
        scale = np.random.uniform(*self.scale_range)
        w, h = image.size
        new_h, new_w = int(h * scale), int(w * scale)
        image = TF.resize(image, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [new_h, new_w], interpolation=TF.InterpolationMode.NEAREST)

        # Pad if smaller than crop_size
        pad_h = max(self.crop_size - new_h, 0)
        pad_w = max(self.crop_size - new_w, 0)
        if pad_h > 0 or pad_w > 0:
            image = TF.pad(image, [0, 0, pad_w, pad_h], fill=0)
            mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=0)

        # Random crop
        i, j, th, tw = transforms.RandomCrop.get_params(image, (self.crop_size, self.crop_size))
        image = TF.crop(image, i, j, th, tw)
        mask = TF.crop(mask, i, j, th, tw)

        # Random horizontal flip
        if np.random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # To tensor + normalize
        image = TF.to_tensor(image)
        image = self.normalize(image)

        mask = torch.from_numpy(np.array(mask)).long()
        # ADE20K: 0 = unlabeled → IGNORE_INDEX, 1-150 → 0-149
        mask[mask == 0] = cfg.IGNORE_INDEX + 1  # temp placeholder
        mask = mask - 1
        mask[mask == cfg.IGNORE_INDEX] = cfg.IGNORE_INDEX  # was 0, now mapped correctly

        return image, mask


class SegValTransform:
    """Validation transform: resize to fixed size, normalize."""

    def __init__(self, size=None):
        self.size = size or cfg.IMG_SIZE
        self.normalize = transforms.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD)

    def __call__(self, image, mask):
        image = TF.resize(image, [self.size, self.size], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.size, self.size], interpolation=TF.InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        image = self.normalize(image)

        mask = torch.from_numpy(np.array(mask)).long()
        # ADE20K label shift: 0 → IGNORE_INDEX, 1-150 → 0-149
        mask[mask == 0] = cfg.IGNORE_INDEX + 1
        mask = mask - 1
        mask[mask == cfg.IGNORE_INDEX] = cfg.IGNORE_INDEX

        return image, mask


def get_train_dataset():
    return ADE20KDataset(cfg.DATA_ROOT, split="training", transform=SegTrainTransform())


def get_val_dataset():
    return ADE20KDataset(cfg.DATA_ROOT, split="validation", transform=SegValTransform())


def get_train_loader(batch_size=None, num_workers=None):
    ds = get_train_dataset()
    nw = num_workers if num_workers is not None else cfg.NUM_WORKERS
    return DataLoader(
        ds,
        batch_size=batch_size or cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
        persistent_workers=nw > 0,
        multiprocessing_context="forkserver" if nw > 0 else None,
    )


def get_val_loader(batch_size=None, num_workers=None):
    ds = get_val_dataset()
    nw = num_workers if num_workers is not None else cfg.NUM_WORKERS
    return DataLoader(
        ds,
        batch_size=batch_size or cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        drop_last=False,
        persistent_workers=nw > 0,
        multiprocessing_context="forkserver" if nw > 0 else None,
    )
