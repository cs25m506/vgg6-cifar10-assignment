"""
train.py  –  Generic training entry-point
==========================================
Train VGG6 on CIFAR-10 for one configuration.
All hyper-parameters can be overridden via CLI flags.

Examples
--------
# Q1 baseline
python train.py

# Q2(a) – change activation
python train.py --activation gelu

# Q2(b) – change optimizer
python train.py --optimizer nesterov_sgd --lr 0.01

# Q2(c) – change hyper-parameters
python train.py --batch_size 64 --num_epochs 50 --lr 5e-4
"""

import argparse
import os
import sys

import wandb  # type: ignore

sys.path.insert(0, os.path.dirname(__file__))

from configs import BASE_CONFIG
from src import (
    VGG6,
    build_optimizer,
    evaluate,
    get_dataloaders,
    get_device,
    save_checkpoint,
    set_seed,
    train_one_epoch,
)
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VGG6 on CIFAR-10")
    p.add_argument("--activation",     type=str,   default=BASE_CONFIG["activation"])
    p.add_argument("--optimizer",      type=str,   default=BASE_CONFIG["optimizer"])
    p.add_argument("--lr",             type=float, default=BASE_CONFIG["lr"])
    p.add_argument("--weight_decay",   type=float, default=BASE_CONFIG["weight_decay"])
    p.add_argument("--batch_size",     type=int,   default=BASE_CONFIG["batch_size"])
    p.add_argument("--num_epochs",     type=int,   default=BASE_CONFIG["num_epochs"])
    p.add_argument("--seed",           type=int,   default=BASE_CONFIG["seed"])
    p.add_argument("--data_dir",       type=str,   default=BASE_CONFIG["data_dir"])
    p.add_argument("--checkpoint_dir", type=str,   default=BASE_CONFIG["checkpoint_dir"])
    p.add_argument("--wandb_project",  type=str,   default=BASE_CONFIG["wandb_project"])
    p.add_argument("--wandb_entity",   type=str,   default=None,
                   help="W&B entity (username or team). If omitted, wandb uses default.")
    p.add_argument("--run_name",       type=str,   default=None,
                   help="Optional W&B run name; auto-generated if omitted.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args   = parse_args()
    config = vars(args)

    # Auto run name
    if config["run_name"] is None:
        config["run_name"] = (
            f"act={config['activation']}_opt={config['optimizer']}"
            f"_bs={config['batch_size']}_lr={config['lr']}_ep={config['num_epochs']}"
        )

    # ── Reproducibility ───────────────────────────────────────────────────
    set_seed(config["seed"])
    device = get_device()

    # ── Data ─────────────────────────────────────────────────────────────
    train_loader, test_loader = get_dataloaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model     = VGG6(num_classes=10, activation=config["activation"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model.parameters(),
        name=config["optimizer"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"]
    )

    # ── W&B init ─────────────────────────────────────────────────────────
    wandb_init_kwargs = dict(
        project=config["wandb_project"],
        name=config["run_name"],
        config={k: v for k, v in config.items()
                if k not in ("wandb_entity", "wandb_project", "run_name")},
    )
    if config.get("wandb_entity"):
        wandb_init_kwargs["entity"] = config["wandb_entity"]

    run = wandb.init(**wandb_init_kwargs)
    print(f"W&B run URL : {run.url}")

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_acc   = 0.0
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    best_ckpt_path = os.path.join(
        config["checkpoint_dir"], f"{config['run_name']}_best.pth"
    )

    for epoch in range(1, config["num_epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, run
        )
        val_loss, val_acc = evaluate(
            model, test_loader, criterion, device, epoch,
            prefix="val", wandb_run=run,
        )
        scheduler.step()

        print(
            f"Epoch [{epoch:03d}/{config['num_epochs']}]  "
            f"Train loss: {tr_loss:.4f}  acc: {tr_acc:.2f}%  |  "
            f"Val loss: {val_loss:.4f}  acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                {
                    "epoch":     epoch,
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_acc":   val_acc,
                    "config":    config,
                },
                best_ckpt_path,
            )
            print(f"  ↑ New best val acc: {best_val_acc:.2f}%  (saved → {best_ckpt_path})")

    print(f"\nTraining complete.  Best val acc = {best_val_acc:.2f}%")
    run.summary["best_val_acc"] = best_val_acc
    run.finish()


if __name__ == "__main__":
    main()
