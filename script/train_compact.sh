#!/bin/bash

# Training script for Compact U-Net (2M parameters) with wandb logging
# This script trains a lightweight 3D U-Net model for Alzheimer's classification

# Configuration
MODEL="compact"
BASE_FEATURES=24
EPOCHS=200
BATCH_SIZE=16
LEARNING_RATE=5e-5
WEIGHT_DECAY=1e-5
TARGET_SHAPE="96 96 96"

# Data paths
TRAIN_DIR="data/processed_oasis/train"
VAL_DIR="data/processed_oasis/val"

# Output paths
CHECKPOINT_DIR="checkpoints/compact_2m"
LOG_DIR="logs/compact_2m_newLRStat"
MODEL_NAME="compact_2m_params_newLRStat"

# Wandb settings
USE_WANDB="--use_wandb"

# Loss function: crossentropy, bce, or focal
LOSS_FN="crossentropy"

# Optional: Use class weights for imbalanced data
# Uncomment the line below to enable class weights
CLASS_WEIGHTS="--use_class_weights"

echo "================================================"
echo "Training Compact U-Net Model (2M parameters)"
echo "================================================"
echo "Model: $MODEL"
echo "Base features: $BASE_FEATURES"
echo "Total epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Loss function: $LOSS_FN"
echo "Wandb: Enabled"
echo "================================================"

# Create directories if they don't exist
mkdir -p $CHECKPOINT_DIR
mkdir -p $LOG_DIR

# Run training
python train.py \
    --train_dir $TRAIN_DIR \
    --val_dir $VAL_DIR \
    --model $MODEL \
    --base_features $BASE_FEATURES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --target_shape $TARGET_SHAPE \
    --loss_fn $LOSS_FN \
    --checkpoint_dir $CHECKPOINT_DIR \
    --log_dir $LOG_DIR \
    --model_name $MODEL_NAME \
    $USE_WANDB \
    $CLASS_WEIGHTS \
    --num_workers 4 \
    --device cuda

echo "================================================"
echo "Training completed!"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo "Logs saved to: $LOG_DIR"
echo "================================================"
