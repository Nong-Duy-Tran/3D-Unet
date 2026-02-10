#!/bin/bash

#############################################################################
# Train a single fold for 5-Fold Cross-Validation
#############################################################################
# Usage: bash script/train_cv5_fold.sh <fold_num> [OPTIONS]
# Example: bash script/train_cv5_fold.sh 0 --model unet --epochs 200
#############################################################################

# Check if fold number is provided
if [ -z "$1" ]; then
    echo "Error: Fold number required"
    echo "Usage: bash script/train_cv5_fold.sh <fold_num> [OPTIONS]"
    echo "Example: bash script/train_cv5_fold.sh 0 --model unet --epochs 200"
    exit 1
fi

FOLD_NUM=$1
shift  # Remove fold number from arguments

# Validate fold number
if ! [[ "$FOLD_NUM" =~ ^[0-4]$ ]]; then
    echo "Error: Fold number must be 0, 1, 2, 3, or 4"
    exit 1
fi

echo "======================================================================"
echo "Training Fold ${FOLD_NUM} - 5-Fold Cross-Validation"
echo "======================================================================"

# Configuration
DATA_ROOT="./data/processed_oasis_cv5"
FOLD_DIR="${DATA_ROOT}/fold_${FOLD_NUM}"
TRAIN_DIR="${FOLD_DIR}/train"
VAL_DIR="${FOLD_DIR}/val"  # Use dedicated validation set
CHECKPOINT_DIR="./checkpoints/cv5_fold_${FOLD_NUM}"
LOG_DIR="./logs/cv5_fold_${FOLD_NUM}"

# Check if data exists
if [ ! -d "$FOLD_DIR" ]; then
    echo "Error: Fold directory not found: $FOLD_DIR"
    echo "Please run preprocessing first:"
    echo "  bash script/preprocess_cv5.sh"
    exit 1
fi

# Check if train and val directories exist
if [ ! -d "$TRAIN_DIR" ]; then
    echo "Error: Training directory not found: $TRAIN_DIR"
    exit 1
fi

if [ ! -d "$VAL_DIR" ]; then
    echo "Error: Validation directory not found: $VAL_DIR"
    exit 1
fi

# Create output directories
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"

echo ""
echo "Configuration:"
echo "  Fold:            ${FOLD_NUM}"
echo "  Training data:   $TRAIN_DIR"
echo "  Validation data: $VAL_DIR"
echo "  Checkpoints:     $CHECKPOINT_DIR"
echo "  Logs:            $LOG_DIR"
echo ""

# Count training samples
TRAIN_NORMAL=$(find "$TRAIN_DIR/normal" -name "*.nii.gz" 2>/dev/null | wc -l)
TRAIN_ALZHEIMER=$(find "$TRAIN_DIR/alzheimer" -name "*.nii.gz" 2>/dev/null | wc -l)
VAL_NORMAL=$(find "$VAL_DIR/normal" -name "*.nii.gz" 2>/dev/null | wc -l)
VAL_ALZHEIMER=$(find "$VAL_DIR/alzheimer" -name "*.nii.gz" 2>/dev/null | wc -l)

echo "Dataset size:"
echo "  Training:   ${TRAIN_NORMAL} normal + ${TRAIN_ALZHEIMER} alzheimer = $((TRAIN_NORMAL + TRAIN_ALZHEIMER)) total"
echo "  Validation: ${VAL_NORMAL} normal + ${VAL_ALZHEIMER} alzheimer = $((VAL_NORMAL + VAL_ALZHEIMER)) total"
echo ""

# Default training parameters (can be overridden)
MODEL="compact"
EPOCHS=200
BATCH_SIZE=4
LR=0.001
BASE_FEATURES=24
TARGET_SHAPE="128 128 128"
USE_CLASS_WEIGHTS="--use_class_weights"
LOSS_FN="crossentropy"
USE_WANDB=""
SEED=42

# Parse additional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
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
        --base_features)
            BASE_FEATURES="$2"
            shift 2
            ;;
        --target_shape)
            TARGET_SHAPE="$2 $3 $4"
            shift 4
            ;;
        --loss_fn)
            LOSS_FN="$2"
            shift 2
            ;;
        --no_class_weights)
            USE_CLASS_WEIGHTS=""
            shift
            ;;
        --use_wandb)
            USE_WANDB="--use_wandb"
            shift
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run 'python train.py --help' for available options"
            shift
            ;;
    esac
done

echo "Training parameters:"
echo "  Model:          $MODEL"
echo "  Epochs:         $EPOCHS"
echo "  Batch size:     $BATCH_SIZE"
echo "  Learning rate:  $LR"
echo "  Base feature:   $BASE_FEATURES"
echo "  Target shape:   $TARGET_SHAPE"
echo "  Loss function:  $LOSS_FN"
echo "  Class weights:  $([ -n "$USE_CLASS_WEIGHTS" ] && echo 'enabled' || echo 'disabled')"
echo "  W&B logging:    $([ -n "$USE_WANDB" ] && echo 'enabled' || echo 'disabled')"
echo "  Random seed:    $SEED"
echo ""

echo "======================================================================"
echo "Starting training..."
echo "======================================================================"
echo ""

# Run training
python3 train.py \
    --train_dir "$TRAIN_DIR" \
    --val_dir "$VAL_DIR" \
    --model "$MODEL" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --base_features $BASE_FEATURES \
    --target_shape $TARGET_SHAPE \
    --loss_fn "$LOSS_FN" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --model_name "${MODEL}_fold${FOLD_NUM}" \
    --log_dir "$LOG_DIR" \
    --seed $SEED \
    $USE_CLASS_WEIGHTS \
    $USE_WANDB

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ Training completed successfully for Fold ${FOLD_NUM}!"
    echo "======================================================================"
    echo ""
    echo "Model saved to: ${CHECKPOINT_DIR}/${MODEL}_fold${FOLD_NUM}.pth"
    echo "Logs saved to:  ${LOG_DIR}"
    echo ""
    echo "Next steps:"
    echo "  1. Train other folds: bash script/train_cv5_fold.sh <0-4>"
    echo "  2. Evaluate this fold: bash script/evaluate_cv5_fold.sh ${FOLD_NUM}"
    echo "  3. Evaluate all folds: bash script/evaluate_cv5_all.sh"
    echo ""
else
    echo ""
    echo "======================================================================"
    echo "✗ Training failed for Fold ${FOLD_NUM}!"
    echo "======================================================================"
    exit 1
fi
