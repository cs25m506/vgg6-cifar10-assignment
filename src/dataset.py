"""
dataset.py  –  CIFAR-10 data pipeline
======================================
Applies the standard training augmentations required by Q1(a):
  • RandomHorizontalFlip
  • RandomCrop(32, padding=4)
  • Normalize(mean, std)  ← computed from the CIFAR-10 training split

Validation / test split uses only Resize + Normalize (no augmentation).
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple

# CIFAR-10 per-channel statistics (pre-computed on the training set)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_transforms(augment: bool = True) -> transforms.Compose:
    """
    Return a torchvision transform pipeline.

    Parameters
    ----------
    augment : bool
        If True  → training pipeline (flip + crop + normalize).
        If False → evaluation pipeline (normalize only).
    """
    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Download (if needed) and return train / test DataLoaders for CIFAR-10.

    Returns
    -------
    train_loader, test_loader
    """
    train_ds = datasets.CIFAR10(
        root=data_dir, train=True,  download=True, transform=get_transforms(augment=True)
    )
    test_ds  = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=get_transforms(augment=False)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    tr, te = get_dataloaders()
    imgs, labels = next(iter(tr))
    print(f"Train batch – images: {imgs.shape}, labels: {labels.shape}")
    imgs, labels = next(iter(te))
    print(f"Test  batch – images: {imgs.shape}, labels: {labels.shape}")
