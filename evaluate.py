"""
evaluate.py  –  Load a saved checkpoint and report test accuracy
================================================================
Usage
-----
python evaluate.py --checkpoint checkpoints/<run_name>_best.pth
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from src import VGG6, get_dataloaders, evaluate as eval_epoch, set_seed, get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved VGG6 checkpoint on CIFAR-10 test set")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint file")
    p.add_argument("--data_dir",   type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = get_device()

    # Load checkpoint
    ckpt   = load_checkpoint(args.checkpoint, device)
    config = ckpt["config"]

    set_seed(config.get("seed", 42))

    # Build model and restore weights
    model = VGG6(num_classes=10, activation=config["activation"]).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from: {args.checkpoint}")
    print(f"  epoch = {ckpt['epoch']}, saved val_acc = {ckpt['val_acc']:.2f}%")

    # Data
    _, test_loader = get_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size)

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    _, test_acc = eval_epoch(model, test_loader, criterion, device, epoch=0)
    print(f"\nTest top-1 accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
