"""
engine.py  –  Training & evaluation loops
==========================================
Provides clean, reusable functions for one epoch of training or evaluation.
W&B logging is optional – pass ``wandb_run=None`` to disable it.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import wandb   # type: ignore


# ---------------------------------------------------------------------------
# Single training epoch
# ---------------------------------------------------------------------------
def train_one_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    criterion:   nn.Module,
    optimizer:   torch.optim.Optimizer,
    device:      torch.device,
    epoch:       int,
    wandb_run:   Optional[wandb.sdk.wandb_run.Run] = None,
) -> tuple[float, float]:
    """
    Run one training epoch.

    Returns
    -------
    avg_loss : float
    top1_acc : float  (0–100)
    """
    model.train()
    running_loss = correct = total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted  = outputs.max(1)
        total        += labels.size(0)
        correct      += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    acc      = 100.0 * correct / total

    if wandb_run is not None:
        wandb_run.log({"train/loss": avg_loss, "train/acc": acc, "epoch": epoch})

    return avg_loss, acc


# ---------------------------------------------------------------------------
# Single evaluation epoch
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    device:     torch.device,
    epoch:      int,
    prefix:     str = "val",
    wandb_run:  Optional[wandb.sdk.wandb_run.Run] = None,
) -> tuple[float, float]:
    """
    Run one evaluation epoch.

    Returns
    -------
    avg_loss : float
    top1_acc : float  (0–100)
    """
    model.eval()
    running_loss = correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted  = outputs.max(1)
        total        += labels.size(0)
        correct      += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    acc      = 100.0 * correct / total

    if wandb_run is not None:
        wandb_run.log({f"{prefix}/loss": avg_loss, f"{prefix}/acc": acc, "epoch": epoch})

    return avg_loss, acc
