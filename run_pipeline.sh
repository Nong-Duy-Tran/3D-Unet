#!/bin/bash
# Quick start script for ADNI Alzheimer's Disease Classification

set -e  # Exit on error

echo "=========================================="
echo "ADNI Alzheimer's Disease Classification"
echo "Quick Start Pipeline"
echo "=========================================="
echo ""

# Configuration
DATA_DIR="./data/datasets/muhammadzahraan/3d-mri-scans-for-alzheimer-disease/versions/4"
OUTPUT_DIR="./data/processed"
CHECKPOINT_DIR="./checkpoints"
LOG_DIR="./logs"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory not found: $DATA_DIR"
    echo "Please update DATA_DIR in this script"
    exit 1
fi

echo "📁 Data directory: $DATA_DIR"
echo "📁 Output directory: $OUTPUT_DIR"
echo ""

# Step 1: Preprocess data
echo "=========================================="
echo "Step 1: Preprocessing Data"
echo "=========================================="
echo ""

if [ -d "$OUTPUT_DIR/train" ] && [ -d "$OUTPUT_DIR/val" ] && [ -d "$OUTPUT_DIR/test" ]; then
    echo "✅ Processed data already exists in $OUTPUT_DIR"
    read -p "Do you want to re-preprocess? (y/N): " response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Skipping preprocessing..."
    else
        echo "Preprocessing data..."
        python preprocess_adni_data.py \
            --data_dir "$DATA_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --train_ratio 0.7 \
            --val_ratio 0.15 \
            --test_ratio 0.15 \
            --seed 42
    fi
else
    echo "Preprocessing data..."
    python preprocess_adni_data.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --train_ratio 0.7 \
        --val_ratio 0.15 \
        --test_ratio 0.15 \
        --seed 42
fi

echo ""
echo "✅ Preprocessing completed!"
echo ""

# Step 2: Train model
echo "=========================================="
echo "Step 2: Training Model"
echo "=========================================="
echo ""

read -p "Do you want to start training? (Y/n): " response
if [[ "$response" =~ ^[Nn]$ ]]; then
    echo "Skipping training. You can train manually with:"
    echo "  python train.py --train_dir $OUTPUT_DIR/train --val_dir $OUTPUT_DIR/val"
    exit 0
fi

echo "Starting training..."
echo "💡 Tip: Monitor training with: tensorboard --logdir $LOG_DIR"
echo ""

python train.py \
    --train_dir "$OUTPUT_DIR/train" \
    --val_dir "$OUTPUT_DIR/val" \
    --model simple \
    --base_features 32 \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4 \
    --target_shape 64 64 64 \
    --use_class_weights \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --log_dir "$LOG_DIR" \
    --num_workers 4

echo ""
echo "✅ Training completed!"
echo ""

# Step 3: Evaluate model
echo "=========================================="
echo "Step 3: Evaluating Model"
echo "=========================================="
echo ""

if [ ! -f "$CHECKPOINT_DIR/best_model.pth" ]; then
    echo "❌ Error: No trained model found at $CHECKPOINT_DIR/best_model.pth"
    echo "Please train the model first."
    exit 1
fi

read -p "Do you want to evaluate the model on test set? (Y/n): " response
if [[ "$response" =~ ^[Nn]$ ]]; then
    echo "Skipping evaluation."
    exit 0
fi

echo "Evaluating model..."
python evaluate.py \
    --checkpoint "$CHECKPOINT_DIR/best_model.pth" \
    --data_dir "$OUTPUT_DIR/test" \
    --model simple \
    --base_features 32 \
    --batch_size 4 \
    --target_shape 64 64 64 \
    --save_dir ./evaluation_results

echo ""
echo "✅ Evaluation completed!"
echo ""

echo "=========================================="
echo "Pipeline Completed Successfully!"
echo "=========================================="
echo ""
echo "📊 Results:"
echo "  - Training logs: $LOG_DIR"
echo "  - Model checkpoint: $CHECKPOINT_DIR/best_model.pth"
echo "  - Evaluation results: ./evaluation_results"
echo ""
echo "📈 To view training curves:"
echo "  tensorboard --logdir $LOG_DIR"
echo ""
