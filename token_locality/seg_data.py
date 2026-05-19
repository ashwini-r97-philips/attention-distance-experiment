"""ADE20K data loading for token locality segmentation experiments.

Copied and made self-contained from bimodal_head_specialisation/segmentation/data.py.
"""

import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

# ─── Constants ────────────────────────────────────────────────────────────────
IGNORE_INDEX  = 255
IMG_SIZE      = 512
CROP_SIZE     = 512
SCALE_RANGE   = (0.5, 2.0)
BATCH_SIZE    = 4
NUM_WORKERS   = 0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class ADE20KDataset(Dataset):
    """ADE20K semantic segmentation dataset.

    ADE20K labels: 0 = unlabeled, 1-150 = classes.
    Shifted here to: 0-149 = classes, IGNORE_INDEX = unlabeled.
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
        mask = Image.open(self.annotations[idx])
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask


class SegTrainTransform:
    def __init__(self, crop_size=CROP_SIZE, scale_range=SCALE_RANGE):
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, image, mask):
        scale = np.random.uniform(*self.scale_range)
        w, h = image.size
        new_h, new_w = int(h * scale), int(w * scale)
        image = TF.resize(image, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [new_h, new_w], interpolation=TF.InterpolationMode.NEAREST)

        pad_h = max(self.crop_size - new_h, 0)
        pad_w = max(self.crop_size - new_w, 0)
        if pad_h > 0 or pad_w > 0:
            image = TF.pad(image, [0, 0, pad_w, pad_h], fill=0)
            mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=0)

        i, j, th, tw = transforms.RandomCrop.get_params(
            image, (self.crop_size, self.crop_size)
        )
        image = TF.crop(image, i, j, th, tw)
        mask = TF.crop(mask, i, j, th, tw)

        if np.random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image = TF.to_tensor(image)
        image = self.normalize(image)

        mask = torch.from_numpy(np.array(mask)).long()
        mask[mask == 0] = IGNORE_INDEX + 1  # temp: protect 0
        mask = mask - 1
        mask[mask == IGNORE_INDEX] = IGNORE_INDEX
        return image, mask


class SegValTransform:
    def __init__(self, size=IMG_SIZE):
        self.size = size
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, image, mask):
        image = TF.resize(image, [self.size, self.size],
                          interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.size, self.size],
                         interpolation=TF.InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        image = self.normalize(image)

        mask = torch.from_numpy(np.array(mask)).long()
        mask[mask == 0] = IGNORE_INDEX + 1
        mask = mask - 1
        mask[mask == IGNORE_INDEX] = IGNORE_INDEX
        return image, mask


def get_train_loader(data_root, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    ds = ADE20KDataset(data_root, split="training", transform=SegTrainTransform())
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        multiprocessing_context="forkserver" if num_workers > 0 else None,
    )


def get_val_loader(data_root, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    ds = ADE20KDataset(data_root, split="validation", transform=SegValTransform())
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
        multiprocessing_context="forkserver" if num_workers > 0 else None,
    )
