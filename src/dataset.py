"""
ENA24 dataset loading.

Assumes images have already been organized into an ImageFolder-compatible
layout:

    data/ena24/<class_name>/<image>.jpg
    data/ena24/<class_name>/<image>.jpg
    ...

If you downloaded the raw LILA BC release (images/ + a COCO-style
annotations JSON), run utils/prepare_ena24.py first to convert it into
this layout. See that script's docstring for details.
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(resolution: int) -> transforms.Compose:
    """Resize/normalize transform for a given input resolution (e.g. 128, 224, 384)."""
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_ena24(
    data_dir: str,
    resolution: int = 224,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[datasets.ImageFolder, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Load ENA24 from an ImageFolder-style directory and split into train/val.

    Returns:
        full_dataset, train_subset, val_subset
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"{data_dir} does not exist. Download ENA24 and, if needed, run "
            "utils/prepare_ena24.py to convert it into ImageFolder format first."
        )

    transform = build_transforms(resolution)
    full_dataset = datasets.ImageFolder(root=str(data_dir), transform=transform)

    val_size = int(len(full_dataset) * val_fraction)
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    return full_dataset, train_subset, val_subset


def make_loader(dataset, batch_size: int, shuffle: bool = False, num_workers: int = 2) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def sample_timing_subset(val_subset, n: int = 75, seed: int = 42):
    """Pull a fixed small subset (50-100 images) reserved purely for latency timing,
    per the evaluation plan in the proposal."""
    generator = torch.Generator().manual_seed(seed)
    n = min(n, len(val_subset))
    indices = torch.randperm(len(val_subset), generator=generator)[:n].tolist()
    return torch.utils.data.Subset(val_subset, indices)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick check that the dataset loads.")
    parser.add_argument("--data_dir", default="data/ena24")
    parser.add_argument("--resolution", type=int, default=224)
    args = parser.parse_args()

    full_ds, train_ds, val_ds = load_ena24(args.data_dir, args.resolution)
    print(f"Classes: {full_ds.classes}")
    print(f"Total images: {len(full_ds)} | train: {len(train_ds)} | val: {len(val_ds)}")
