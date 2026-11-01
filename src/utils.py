"""
utils.py  –  Reproducibility, checkpointing, and misc helpers
==============================================================
"""

from __future__ import annotations
import os
import random
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """
    Fix all random seeds for reproducibility.
    Call this at the very start of every script/experiment.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic (slightly slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """Return GPU if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[utils] Using device: {device}")
    return device


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(
    state:    dict,
    filepath: str,
) -> None:
    """
    Save a training checkpoint dictionary to *filepath*.

    Typical usage
    -------------
    save_checkpoint({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "val_acc":    val_acc,
        "config":     config,
    }, filepath="checkpoints/best_model.pth")
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath: str, device: torch.device) -> dict:
    """Load and return a checkpoint dictionary from *filepath*."""
    return torch.load(filepath, map_location=device)
