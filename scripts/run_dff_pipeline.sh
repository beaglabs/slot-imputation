#!/bin/bash
set -e

DETECT_DEVICE=$(python3 -c "
import torch
if torch.cuda.is_available():
    print('cuda')
elif torch.backends.mps.is_available():
    print('mps')
else:
    print('cpu')
")

echo "Detected device: $DETECT_DEVICE"

run_tool() {
    python3 -m slot_impute."$@"
}

show() { echo "=== $1 ==="; mkdir -p checkpoints_dff; }

# --- Phase 1: Quick test on tiny model (local validation, ~30s on M2) ---
show "Phase 1: Quick test"
QT_D_FF=$(python3 -c "import yaml; c=yaml.safe_load(open('experiments/dff_variation/config.yaml')); print(c['quick_test']['d_ff'])")
QT_TGT=$(python3 -c "import yaml; c=yaml.safe_load(open('experiments/dff_variation/config.yaml')); print(c['quick_test']['target_d_ff'])")

run_tool train_transformer --config experiments/dff_variation/config.yaml --device "$DETECT_DEVICE" --save-dir checkpoints_dff --tag quick_test --quick-test
SOURCE_PATH="checkpoints_dff/dff${QT_D_FF}_seed42_quick_test.pt"

run_tool validate_mlp --config experiments/dff_variation/config.yaml --source-path "$SOURCE_PATH" --target-d-ff "$QT_TGT" --device "$DETECT_DEVICE" --output quick_test_results.json

IMPUTED_PPL=$(python3 -c "import json; r=json.load(open('quick_test_results.json')); print(r['imputed_ppl'])")
echo "Quick test imputed PPL: $IMPUTED_PPL"
if python3 -c "exit(0 if $IMPUTED_PPL < 10000 else 1)"; then
    echo "Quick test PASSED (PPL=$IMPUTED_PPL < 10000). Signal is real."
else
    echo "Quick test FAILED (PPL=$IMPUTED_PPL). Review logs before proceeding."
    exit 1
fi

if [ "$1" == "--quick-only" ]; then
    echo "Exiting (--quick-only). Quick test confirmed."
    exit 0
fi

# --- Phase 2: Full training ---
show "Phase 2: Full training"
FULL_D_FF=$(python3 -c "import yaml; c=yaml.safe_load(open('experiments/dff_variation/config.yaml')); print(c['d_ff_variation']['source_d_ff'])")
FULL_TGT=$(python3 -c "import yaml; c=yaml.safe_load(open('experiments/dff_variation/config.yaml')); print(c['d_ff_variation']['target_d_ff'])")
FULL_STEPS=$(python3 -c "import yaml; c=yaml.safe_load(open('experiments/dff_variation/config.yaml')); print(c['training']['steps'])")

run_tool train_transformer --config experiments/dff_variation/config.yaml --d-ff "$FULL_D_FF" --seed 42 --steps "$FULL_STEPS" --device "$DETECT_DEVICE" --save-dir checkpoints_dff --tag source

run_tool train_transformer --config experiments/dff_variation/config.yaml --d-ff "$FULL_TGT" --seed 42 --steps "$FULL_STEPS" --device "$DETECT_DEVICE" --save-dir checkpoints_dff --tag ground_truth

SOURCE_PATH="checkpoints_dff/dff${FULL_D_FF}_seed42_source.pt"
GT_PATH="checkpoints_dff/dff${FULL_TGT}_seed42_ground_truth.pt"

# --- Phase 3: Validate ---
show "Phase 3: Full validation"
run_tool validate_mlp --config experiments/dff_variation/config.yaml --source-path "$SOURCE_PATH" --ground-truth-path "$GT_PATH" --target-d-ff "$FULL_TGT" --device "$DETECT_DEVICE" --output dff_validation_results.json

echo "=== Done. See dff_validation_results.json ==="