#!/usr/bin/env bash
# Q1 – Best single configuration baseline
# Usage: bash scripts/run_q1_baseline.sh [--wandb_entity <username>]
set -e
ENTITY="${WANDB_ENTITY:-}"   # set WANDB_ENTITY env var or pass --wandb_entity

python train.py \
    --activation   relu   \
    --optimizer    adam   \
    --lr           1e-3   \
    --batch_size   128    \
    --num_epochs   30     \
    --seed         42     \
    ${ENTITY:+--wandb_entity "$ENTITY"} \
    --run_name     "Q1_baseline_relu_adam"
