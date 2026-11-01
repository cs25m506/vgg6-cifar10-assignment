#!/usr/bin/env bash
# Q2(b) – Optimizer sweep (6 runs)
# SGD-family: lr=0.01 | Adaptive: lr=1e-3
set -e
ENTITY="${WANDB_ENTITY:-}"

declare -A OPT_LR=(
    ["sgd"]=0.01
    ["nesterov_sgd"]=0.01
    ["adam"]=1e-3
    ["adagrad"]=1e-2
    ["rmsprop"]=1e-3
    ["nadam"]=1e-3
)

for OPT in sgd nesterov_sgd adam adagrad rmsprop nadam; do
    LR="${OPT_LR[$OPT]}"
    echo "\n===== Q2b: optimizer = $OPT  lr = $LR ====="
    python train.py \
        --activation   relu   \
        --optimizer    "$OPT" \
        --lr           "$LR"  \
        --batch_size   128    \
        --num_epochs   30     \
        --seed         42     \
        ${ENTITY:+--wandb_entity "$ENTITY"} \
        --run_name     "Q2b_opt=$OPT"
done
