"""
base_config.py  –  Default hyper-parameters (Q1 baseline)
==========================================================
All experiment scripts import from here.  Override any key via
command-line args or sweep configs.
"""

BASE_CONFIG = {
    # Data
    "data_dir":     "./data",
    "num_workers":  2,

    # Architecture
    "num_classes":  10,
    "activation":   "relu",           # best single-config baseline

    # Optimisation
    "optimizer":    "adam",
    "lr":           1e-3,
    "weight_decay": 5e-4,
    "batch_size":   128,
    "num_epochs":   30,

    # Reproducibility
    "seed": 42,

    # Checkpointing
    "checkpoint_dir": "./checkpoints",

    # W&B project name
    "wandb_project": "CS6886W-A1-VGG6-CIFAR10",
}
