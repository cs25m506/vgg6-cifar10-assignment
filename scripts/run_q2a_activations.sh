#!/usr/bin/env bash
# Q2(a) – Activation sweep (5 runs)
set -e
ENTITY="${WANDB_ENTITY:-}"

for ACT in relu sigmoid tanh silu gelu; do
    echo "\n===== Q2a: activation = $ACT ====="
    python train.py \
        --activation   "$ACT" \
        --optimizer    adam   \
        --lr           1e-3   \
        --batch_size   128    \
        --num_epochs   30     \
        --seed         42     \
        ${ENTITY:+--wandb_entity "$ENTITY"} \
        --run_name     "Q2a_act=$ACT"
done
