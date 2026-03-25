#!/bin/bash
# Training script for all 5 folds of 2D CNN + Attention model

echo "=========================================="
echo "2D CNN + Attention Training - All Folds"
echo "=========================================="

# Default parameters
DATA_DIR="../data/processed_oasis_2d_cv5_coronal"
EPOCHS=100
BATCH_SIZE=16
LR=1e-4
WEIGHT_DECAY=1e-4

# Model parameters
MODEL_NAME="standard"
BASE_CHANNELS=16
DROPOUT=0.3
NUM_SLICES=80

# Training parameters
OPTIMIZER="adamw"
SCHEDULER="cosine"
USE_CLASS_WEIGHTS=false
USE_AMP=false
USE_WANDB=true
WANDB_GROUP="coronal-remove-block4-base16"

# Directories
CHECKPOINT_DIR="../checkpoints/coronal_remove_block4/base_16"
LOG_DIR="../logs/2d_cv5_sample"

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
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo ""
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Model: $MODEL_NAME"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Base channels: $BASE_CHANNELS"
echo "  Use class weights: $USE_CLASS_WEIGHTS"
echo "  Use AMP: $USE_AMP"
echo "  Use WandB: $USE_WANDB"
echo ""

# Run training for all folds
echo "=========================================="
echo "Starting 5-fold cross-validation training..."
echo "=========================================="
echo ""

FAILED_FOLDS=()
cd "$(dirname "$0")/../2d" || exit 1

for FOLD in 0 1 2 3 4; do
    echo "------------------------------------------"
    echo "Training fold $FOLD"
    echo "------------------------------------------"

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

    if [ "$USE_CLASS_WEIGHTS" = true ]; then
        CMD="$CMD --use_class_weights"
    fi

    if [ "$USE_AMP" = true ]; then
        CMD="$CMD --use_amp"
    fi

    if [ "$USE_WANDB" = true ]; then
        CMD="$CMD --use_wandb --wandb_project alzheimer-2d-classification --exp_name 2d --wandb_group $WANDB_GROUP"
    fi

    eval $CMD
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "Fold $FOLD failed with exit code $EXIT_CODE"
        FAILED_FOLDS+=("$FOLD")
    fi

    echo ""
done

if [ ${#FAILED_FOLDS[@]} -eq 0 ]; then
    echo "=========================================="
    echo "All folds training completed successfully!"
    echo "=========================================="
else
    echo "=========================================="
    echo "Training completed with failures in folds: ${FAILED_FOLDS[*]}"
    echo "=========================================="
    exit 1
fi
