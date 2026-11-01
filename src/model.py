"""
model.py  –  VGG6 architecture for CIFAR-10
============================================
VGG6 block layout
  Conv(64)  → Conv(64)  → MaxPool
  Conv(128) → Conv(128) → MaxPool
  FC(512) → FC(512) → FC(num_classes)

The activation function is injected at construction time so the same
class can be used for all Question-2(a) activation experiments.
"""

import torch
import torch.nn as nn
from typing import Type

# ---------------------------------------------------------------------------
# Supported activation factories
# ---------------------------------------------------------------------------
ACTIVATIONS: dict[str, Type[nn.Module]] = {
    "relu":    nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh":    nn.Tanh,
    "silu":    nn.SiLU,
    "gelu":    nn.GELU,
}


def get_activation(name: str) -> nn.Module:
    """Return an activation module by name (case-insensitive)."""
    key = name.lower()
    if key not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(ACTIVATIONS)}")
    return ACTIVATIONS[key]()


# ---------------------------------------------------------------------------
# VGG6 model
# ---------------------------------------------------------------------------
class VGG6(nn.Module):
    """
    VGG6 – a lightweight VGG-style network for CIFAR-10.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 10 for CIFAR-10).
    activation  : str
        Activation function to use throughout the network.
        One of 'relu', 'sigmoid', 'tanh', 'silu', 'gelu'.
    """

    def __init__(self, num_classes: int = 10, activation: str = "relu") -> None:
        super().__init__()
        act = lambda: get_activation(activation)   # factory – fresh instance each call

        self.features = nn.Sequential(
            # Block 1 – 3 → 64
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            act(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            act(),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 32 → 16

            # Block 2 – 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            act(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            act(),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 16 → 8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            act(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            act(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 3, 32, 32) → (B, C)
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for act_name in ACTIVATIONS:
        m = VGG6(activation=act_name)
        dummy = torch.randn(2, 3, 32, 32)
        out = m(dummy)
        assert out.shape == (2, 10), f"Bad output shape for activation={act_name}"
        print(f"  VGG6(activation={act_name!r}) output shape: {out.shape}  ✓")
