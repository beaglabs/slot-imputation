#!/bin/bash
# Train all anchor models. Reads config.yaml for M values and seeds.
# Usage: bash scripts/train_anchors.sh [--parallel N]

CONFIG="experiments/m_variation/config.yaml"
CHECKPOINT_DIR="checkpoints"
DEVICE="cpu"

M_VALUES=(128 192 256 384 512)
SEEDS=(42 123 999)

mkdir -p "$CHECKPOINT_DIR"

for M in "${M_VALUES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Training M=$M seed=$SEED..."
        python3 -m slot_impute.model \
            --num-slots $M \
            --seed $SEED \
            --seq-len 2048 \
            --corruption-rate 0.50 \
            --save-dir "$CHECKPOINT_DIR" \
            --device "$DEVICE"
    done
done

echo "Done. Checkpoints in $CHECKPOINT_DIR/"