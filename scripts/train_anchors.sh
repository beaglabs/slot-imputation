#!/bin/bash
# Train all anchor models. Reads config.yaml for M values and seeds.
# Usage: bash scripts/train_anchors.sh [--parallel N]

CONFIG="experiments/m_variation/config.yaml"
CHECKPOINT_DIR="checkpoints"
DEVICE="cpu"

TASK=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('experiment',{}).get('task','random'))")
if [ "$TASK" = "structured" ]; then
    NUM_STATES=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['task']['structured']['num_states'])")
    CHAIN_SEED=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['task']['structured']['chain_seed'])")
    CORR_RATE=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['training']['corruption_rate'])")
    TASK_FLAGS="--task structured --num-states $NUM_STATES --chain-seed $CHAIN_SEED"
else
    TASK_FLAGS="--task random"
    CORR_RATE="0.15"
fi

M_VALUES=(128 192 256 384 512)
SEEDS=(42 123 999)

mkdir -p "$CHECKPOINT_DIR"

for M in "${M_VALUES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Training M=$M seed=$SEED task=$TASK..."
        python3 -m slot_impute.model \
            --num-slots $M \
            --seed $SEED \
            --seq-len 2048 \
            --corruption-rate $CORR_RATE \
            --save-dir "$CHECKPOINT_DIR" \
            --device "$DEVICE" \
            $TASK_FLAGS
    done
done

echo "Done. Checkpoints in $CHECKPOINT_DIR/"