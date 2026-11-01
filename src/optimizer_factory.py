"""
optimizer_factory.py  –  Build any Q2(b) optimizer from a name + LR
=====================================================================
Supported names (case-insensitive):
  sgd, nesterov_sgd, adam, adagrad, rmsprop, nadam
"""

import torch
import torch.optim as optim


SUPPORTED_OPTIMIZERS = [
    "sgd", "nesterov_sgd", "adam", "adagrad", "rmsprop", "nadam"
]


def build_optimizer(
    params,
    name:         str,
    lr:           float = 1e-3,
    weight_decay: float = 5e-4,
) -> optim.Optimizer:
    """
    Construct and return a PyTorch optimizer.

    Parameters
    ----------
    params       : model.parameters()
    name         : optimizer name (see SUPPORTED_OPTIMIZERS)
    lr           : learning rate
    weight_decay : L2 regularisation coefficient
    """
    key = name.lower()

    if key == "sgd":
        return optim.SGD(
            params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=False
        )

    elif key == "nesterov_sgd":
        return optim.SGD(
            params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True
        )

    elif key == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)

    elif key == "adagrad":
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)

    elif key == "rmsprop":
        return optim.RMSprop(
            params, lr=lr, momentum=0.9, weight_decay=weight_decay
        )

    elif key == "nadam":
        return optim.NAdam(params, lr=lr, weight_decay=weight_decay)

    else:
        raise ValueError(
            f"Unknown optimizer '{name}'. "
            f"Choose from: {SUPPORTED_OPTIMIZERS}"
        )
