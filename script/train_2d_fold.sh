#!/bin/bash
# Training script for single fold of 2D CNN + Attention model

echo "=========================================="
echo "2D CNN + Attention Training - Single Fold"
echo "=========================================="

# Default parameters
DATA_DIR="../data/processed_oasis_2d_cv5"
FOLD=0
EPOCHS=100
BATCH_SIZE=4
LR=1e-4
WEIGHT_DECAY=1e-4

# Model parameters
MODEL_NAME="mrinet"
BASE_CHANNELS=24
DROPOUT=0.3
NUM_SLICES=120

# Training parameters
OPTIMIZER="adamw"
SCHEDULER="cosine"
USE_CLASS_WEIGHTS=false
USE_AMP=false
USE_WANDB=false

# Directories
CHECKPOINT_DIR="../checkpoints/2d_cv5"
LOG_DIR="../logs/2d_cv5"

# Other
NUM_WORKERS=4
SEED=42
SAVE_FREQ=20

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
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
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
        --use_class_weights)
            USE_CLASS_WEIGHTS=true
            shift
            ;;
        --use_amp)
            USE_AMP=true
            shift
            ;;
        --use_wandb)
            USE_WANDB=true
            shift
            ;;
        --resume)
            RESUME="--resume"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo ""
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Fold: $FOLD"
echo "  Model: $MODEL_NAME"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Base channels: $BASE_CHANNELS"
echo "  Optimizer: $OPTIMIZER"
echo "  Use class weights: $USE_CLASS_WEIGHTS"
echo "  Use AMP: $USE_AMP"
echo "  Use WandB: $USE_WANDB"
echo ""

# Build command
CMD="python train.py \
    --data_dir $DATA_DIR \
    --fold $FOLD \
    --num_slices $NUM_SLICES \
    --model_name $MODEL_NAME \
    --base_channels $BASE_CHANNELS \
    --dropout $DROPOUT \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --optimizer $OPTIMIZER \
    --scheduler $SCHEDULER \
    --checkpoint_dir $CHECKPOINT_DIR \
    --log_dir $LOG_DIR \
    --num_workers $NUM_WORKERS \
    --seed $SEED \
    --save_freq $SAVE_FREQ"

# Add optional flags
if [ "$USE_CLASS_WEIGHTS" = true ]; then
    CMD="$CMD --use_class_weights"
fi

if [ "$USE_AMP" = true ]; then
    CMD="$CMD --use_amp"
fi

if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --use_wandb --wandb_project alzheimer-2d-classification --exp_name 2d_fold${FOLD}"
fi

if [ ! -z "$RESUME" ]; then
    CMD="$CMD $RESUME"
fi

# Run training
echo "=========================================="
echo "Starting training..."
echo "=========================================="
echo ""

cd "$(dirname "$0")/../2d" || exit 1
eval $CMD

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Training failed with exit code $EXIT_CODE"
    echo "=========================================="
    exit $EXIT_CODE
fi
