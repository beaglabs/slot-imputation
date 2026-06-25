#!/bin/bash
set -e

CONFIG="experiments/dff_variation/config.yaml"
CHECKPOINT_DIR="checkpoints_dff"
DEVICE="cpu"

SOURCE_D_FF=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['d_ff_variation']['source_d_ff'])")
TARGET_D_FF=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['d_ff_variation']['target_d_ff'])")
STEPS=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['training']['steps'])")

mkdir -p "$CHECKPOINT_DIR"

echo "=== Training source model (d_ff=$SOURCE_D_FF) ==="
python3 -m slot_impute.train_transformer \
    --config "$CONFIG" \
    --d-ff "$SOURCE_D_FF" \
    --seed 42 \
    --steps "$STEPS" \
    --device "$DEVICE" \
    --save-dir "$CHECKPOINT_DIR" \
    --tag "source"

echo "=== Training ground truth model (d_ff=$TARGET_D_FF) ==="
python3 -m slot_impute.train_transformer \
    --config "$CONFIG" \
    --d-ff "$TARGET_D_FF" \
    --seed 42 \
    --steps "$STEPS" \
    --device "$DEVICE" \
    --save-dir "$CHECKPOINT_DIR" \
    --tag "ground_truth"

echo "Done. Checkpoints in $CHECKPOINT_DIR/"