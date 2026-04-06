#!/bin/bash
# Train a single fold for 5-Fold Cross-Validation

FOLD=0

# Configuration
DATA_DIR="../data/mediqa"
CHECKPOINT_DIR="../checkpoints/mediqa-vit-unetr"
LOG_DIR="../logs/mediqa-vit-unetr"

# Default training parameters (can be overridden)
MODEL_NAME="vit_unetr"
EPOCHS=400
BATCH_SIZE=8
LR=0.001
BASE_FEATURES=24
TARGET_SHAPE="96 96 96"
LOSS_FN="crossentropy"
USE_WANDB=true
USE_CLASS_WEIGHTS=true
WANDB_GROUP="vit-unetr"
RESUME_CHECKPOINT=""

SEED=42

# Parse additional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fold)
            FOLD="$2"
            shift 2
            ;;
        --mode_namel)
            MODEL_NAME="$2"
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
        --use_class_weights)
            USE_CLASS_WEIGHTS=true
            shift
            ;;
        --use_wandb)
            USE_WANDB=true
            shift
            ;;
        --wandb_group)
            WANDB_GROUP="$2"
            shift 2
            ;;
        --resume)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash script/train_3d_fold.sh --fold <0-4> [OPTIONS]"
            exit 1
            ;;
    esac
done


echo ""
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Fold: $FOLD"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  Logs: $LOG_DIR"
echo "  Model: $MODEL_NAME"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  Base features: $BASE_FEATURES"
echo "  Use class weights: $USE_CLASS_WEIGHTS"
echo "  Use WandB: $USE_WANDB"
echo "  WandB: $WANDB_GROUP"
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "  Resume from: $RESUME_CHECKPOINT"
fi
echo ""

# Run training
CMD="python train.py \
    --data_dir "$DATA_DIR" \
    --fold $FOLD\
    --model_name "$MODEL_NAME" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --base_features $BASE_FEATURES \
    --target_shape $TARGET_SHAPE \
    --loss_fn "$LOSS_FN" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --log_dir "$LOG_DIR" \
    --seed $SEED"

if [ -n "$RESUME_CHECKPOINT" ]; then
    CMD="$CMD --resume \"$RESUME_CHECKPOINT\""
fi

if [ "$USE_CLASS_WEIGHTS" = true ]; then
    CMD="$CMD --use_class_weights"
fi

if [ "$USE_AMP" = true ]; then
    CMD="$CMD --use_amp"
fi

if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --use_wandb --wandb_project mediqa --exp_name 3d_model --wandb_group $WANDB_GROUP"
fi

# Run training
echo "=========================================="
echo "Starting training..."
echo "=========================================="
echo ""

cd "$(dirname "$0")/../3d" || exit 1
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
