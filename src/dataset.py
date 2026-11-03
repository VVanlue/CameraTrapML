"""
ENA24 dataset loading.
"""


import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(resolution: int, train: bool = False) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(resolution, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.2),  # day/night/IR lighting variation
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _group_key(path: str, group_size: int) -> str:
    """Burst-aware group id from the file name: '1234_5678.jpg' -> image id 1234 -> group 123."""
    stem = Path(path).stem
    m = re.match(r"(\d+)", stem)
    if m and group_size > 1:
        return f"g{int(m.group(1)) // group_size}"
    return stem


def split_indices(samples, val_fraction=0.15, test_fraction=0.15, seed=42, group_size=10):
    """Stratified-by-class, grouped split. Returns (train_idx, val_idx, test_idx)."""
    rng = np.random.default_rng(seed)
    by_class = {}
    for i, (path, label) in enumerate(samples):
        by_class.setdefault(label, {}).setdefault(_group_key(path, group_size), []).append(i)

    train, val, test = [], [], []
    for label, groups in by_class.items():
        keys = sorted(groups)
        rng.shuffle(keys)
        n = len(keys)
        n_test = max(1, int(round(n * test_fraction))) if n >= 3 else 0
        n_val = max(1, int(round(n * val_fraction))) if n >= 3 else 0
        for k in keys[:n_test]:
            test += groups[k]
        for k in keys[n_test:n_test + n_val]:
            val += groups[k]
        for k in keys[n_test + n_val:]:
            train += groups[k]
    return sorted(train), sorted(val), sorted(test)


def load_ena24(data_dir: str, resolution: int = 224, seed: int = 42, group_size: int = 10,
               val_fraction: float = 0.15, test_fraction: float = 0.15):
    """
    Returns (classes, train_ds, val_ds, test_ds). train_ds uses augmentation,
    val/test use the plain resize transform.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"{data_dir} does not exist. Run src/prepare_ena24.py first (see README step 2).")

    train_base = datasets.ImageFolder(str(data_dir), transform=build_transforms(resolution, train=True))
    eval_base = datasets.ImageFolder(str(data_dir), transform=build_transforms(resolution, train=False))
    tr, va, te = split_indices(eval_base.samples, val_fraction, test_fraction, seed, group_size)
    return eval_base.classes, Subset(train_base, tr), Subset(eval_base, va), Subset(eval_base, te)


def make_loader(dataset, batch_size: int, shuffle: bool = False, num_workers: int = 2) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      persistent_workers=num_workers > 0)


def fixed_subset(ds, n: int, seed: int = 42) -> Subset:
    """A fixed, reproducible subset (used for calibration and timing)."""
    g = torch.Generator().manual_seed(seed)
    n = min(n, len(ds))
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    return Subset(ds, idx)


def preload_tensors(ds, n: int = 96, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load n images into a single [n,3,H,W] tensor (+labels). n=96 divides evenly
    by batch sizes 1, 2, 4, 8, 16, 32, so every timed batch is full-sized."""
    sub = fixed_subset(ds, n, seed)
    xs, ys = zip(*[sub[i] for i in range(len(sub))])
    return torch.stack(xs).contiguous(), torch.tensor(ys)


def class_counts(ds) -> List[int]:
    base = ds.dataset if isinstance(ds, Subset) else ds
    idx = ds.indices if isinstance(ds, Subset) else range(len(base))
    counts = np.bincount([base.targets[i] for i in idx], minlength=len(base.classes))
    return counts.tolist()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Check that the dataset loads and show the split.")
    ap.add_argument("--data_dir", default="data/ena24_crops")
    ap.add_argument("--resolution", type=int, default=224)
    a = ap.parse_args()
    classes, tr, va, te = load_ena24(a.data_dir, a.resolution)
    print(f"{len(classes)} classes: {classes}")
    print(f"train {len(tr)} | val {len(va)} | test {len(te)}")
    for c, n_tr, n_te in zip(classes, class_counts(tr), class_counts(te)):
        print(f"  {c:25s} train={n_tr:5d} test={n_te:4d}")
