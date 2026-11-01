"""
src  –  VGG6 / CIFAR-10 training package
=========================================
Modules
-------
model              : VGG6 architecture
dataset            : CIFAR-10 data pipeline
engine             : train_one_epoch / evaluate
optimizer_factory  : build_optimizer
utils              : set_seed, get_device, save/load_checkpoint
"""

from .model              import VGG6, get_activation, ACTIVATIONS
from .dataset            import get_dataloaders, get_transforms
from .engine             import train_one_epoch, evaluate
from .optimizer_factory  import build_optimizer, SUPPORTED_OPTIMIZERS
from .utils              import set_seed, get_device, save_checkpoint, load_checkpoint

__all__ = [
    "VGG6", "get_activation", "ACTIVATIONS",
    "get_dataloaders", "get_transforms",
    "train_one_epoch", "evaluate",
    "build_optimizer", "SUPPORTED_OPTIMIZERS",
    "set_seed", "get_device", "save_checkpoint", "load_checkpoint",
]
