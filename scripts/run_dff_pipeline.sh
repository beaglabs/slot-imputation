#!/bin/bash
set -e

CONFIG="experiments/dff_variation/config.yaml"
CHECKPOINT_DIR="checkpoints_dff"
DEVICE="cpu"

SOURCE_D_FF=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['d_ff_variation']['source_d_ff'])")
TARGET_D_FF=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['d_ff_variation']['target_d_ff'])")

echo "=== Phase 1: Quick test ==="
python3 -m slot_impute.train_transformer \
    --config "$CONFIG" \
    --device "$DEVICE" \
    --save-dir "$CHECKPOINT_DIR" \
    --tag "quick_test" \
    --quick-test

SOURCE_PATH="$CHECKPOINT_DIR/dff${SOURCE_D_FF}_seed42_quick_test.pt"

python3 -m slot_impute.validate_mlp \
    --config "$CONFIG" \
    --source-path "$SOURCE_PATH" \
    --device "$DEVICE" \
    --output "quick_test_results.json"

IMPUTED_PPL=$(python3 -c "import json; r=json.load(open('quick_test_results.json')); print(r['imputed_ppl'])")
if python3 -c "exit(0 if $IMPUTED_PPL < 10000 else 1)"; then
    echo "Quick test passed (PPL=$IMPUTED_PPL < 10000). Proceeding to full train."
else
    echo "Quick test FAILED (PPL=$IMPUTED_PPL). Check logs."
    exit 1
fi

echo "=== Phase 2: Full training ==="
bash scripts/train_dff.sh

SOURCE_PATH="$CHECKPOINT_DIR/dff${SOURCE_D_FF}_seed42_source.pt"
GT_PATH="$CHECKPOINT_DIR/dff${TARGET_D_FF}_seed42_ground_truth.pt"

echo "=== Phase 3: Validate ==="
python3 -m slot_impute.validate_mlp \
    --config "$CONFIG" \
    --source-path "$SOURCE_PATH" \
    --ground-truth-path "$GT_PATH" \
    --device "$DEVICE" \
    --output "dff_validation_results.json"

echo "=== Done. See dff_validation_results.json ==="