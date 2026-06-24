#!/bin/bash
# Full pipeline: train -> interferometry -> variogram -> impute -> validate -> report
# Usage: bash scripts/run_pipeline.sh [--skip-train]

set -e

if [ "$1" != "--skip-train" ]; then
    bash scripts/train_anchors.sh
fi

CONFIG="experiments/m_variation/config.yaml"
CHECKPOINT_DIR="checkpoints"

echo "=== Phase 2: Interferometry ==="
python3 -m slot_impute.interferometry --checkpoint-dir "$CHECKPOINT_DIR" --config "$CONFIG"

echo "=== Phase 3: Variogram ==="
python3 -m slot_impute.variogram --checkpoint-dir "$CHECKPOINT_DIR" --config "$CONFIG"

echo "=== Phase 4: Imputation ==="
python3 -m slot_impute.impute --checkpoint-dir "$CHECKPOINT_DIR" --config "$CONFIG" --output-dir imputed/

echo "=== Phase 5: Validation ==="
python3 -m slot_impute.validate --checkpoint-dir "$CHECKPOINT_DIR" --imputed-dir imputed/ --config "$CONFIG"

echo "=== Generate Report ==="
python3 -m slot_impute.report --results validation_results.json --output report.md

echo "=== Done. See report.md ==="