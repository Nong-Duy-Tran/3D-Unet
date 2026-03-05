#!/bin/bash
# Evaluation script for a single fold of 2D CNN + Attention model

echo "=========================================="
echo "2D CNN + Attention Evaluation - Single Fold"
echo "=========================================="

# Default parameters
DATA_DIR="./data/processed_oasis_2d_cv5"
FOLD=0
BATCH_SIZE=4

# Model parameters
MODEL_NAME="standard"
BASE_CHANNELS=24
DROPOUT=0.3
NUM_SLICES=120

# Directories
CHECKPOINT="./checkpoints/2d_cv5/fold_${FOLD}/best_model.pth"
SAVE_DIR="./evaluation_results/2d_cv5"

# Other
NUM_WORKERS=4

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --fold)
            FOLD="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --base_channels)
            BASE_CHANNELS="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --save_dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Default checkpoint uses the resolved fold value
: "${CHECKPOINT:=./checkpoints/2d_cv5/fold_${FOLD}/best_model.pth}"

echo ""
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Fold: $FOLD"
echo "  Model: $MODEL_NAME"
echo "  Batch size: $BATCH_SIZE"
echo "  Base channels: $BASE_CHANNELS"
echo "  Checkpoint: $CHECKPOINT"
echo "  Save directory: $SAVE_DIR"
echo ""

# Run evaluation
echo "=========================================="
echo "Starting evaluation..."
echo "=========================================="
echo ""

python 2d/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --data_dir "$DATA_DIR" \
    --fold "$FOLD" \
    --num_slices "$NUM_SLICES" \
    --model_name "$MODEL_NAME" \
    --base_channels "$BASE_CHANNELS" \
    --dropout "$DROPOUT" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --save_dir "$SAVE_DIR" \
    --seed 42

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Evaluation completed successfully!"
    echo "=========================================="
    echo "Results saved to: $SAVE_DIR"
else
    echo ""
    echo "=========================================="
    echo "Evaluation failed with exit code $EXIT_CODE"
    echo "=========================================="
    exit $EXIT_CODE
fi
