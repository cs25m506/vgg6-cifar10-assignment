#!/usr/bin/env bash
# Q2(c) – Batch size / LR / Epoch sweep (12 runs)
set -e
ENTITY="${WANDB_ENTITY:-}"

echo "\n=== Q2c: Batch size sweep ==="
for BS in 32 64 128 256; do
    python train.py \
        --activation relu  --optimizer adam \
        --lr 1e-3  --batch_size "$BS"  --num_epochs 30 \
        --seed 42  ${ENTITY:+--wandb_entity "$ENTITY"} \
        --run_name "Q2c_bs=$BS"
done

echo "\n=== Q2c: LR sweep ==="
for LR in 1e-4 5e-4 1e-3 5e-3; do
    python train.py \
        --activation relu  --optimizer adam \
        --lr "$LR"  --batch_size 128  --num_epochs 30 \
        --seed 42  ${ENTITY:+--wandb_entity "$ENTITY"} \
        --run_name "Q2c_lr=$LR"
done

echo "\n=== Q2c: Epoch sweep ==="
for EP in 10 20 30 50; do
    python train.py \
        --activation relu  --optimizer adam \
        --lr 1e-3  --batch_size 128  --num_epochs "$EP" \
        --seed 42  ${ENTITY:+--wandb_entity "$ENTITY"} \
        --run_name "Q2c_ep=$EP"
done
