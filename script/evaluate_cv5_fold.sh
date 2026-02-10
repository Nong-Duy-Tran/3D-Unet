#!/bin/bash

#############################################################################
# Evaluate a single fold for 5-Fold Cross-Validation
#############################################################################
# Usage: bash script/evaluate_cv5_fold.sh <fold_num> [OPTIONS]
# Example: bash script/evaluate_cv5_fold.sh 0 --model unet
#############################################################################

# Check if fold number is provided
if [ -z "$1" ]; then
    echo "Error: Fold number required"
    echo "Usage: bash script/evaluate_cv5_fold.sh <fold_num> [OPTIONS]"
    echo "Example: bash script/evaluate_cv5_fold.sh 0 --model unet"
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
echo "Evaluating Fold ${FOLD_NUM} - 5-Fold Cross-Validation"
echo "======================================================================"

# Configuration
DATA_ROOT="./data/processed_oasis_cv5"
FOLD_DIR="${DATA_ROOT}/fold_${FOLD_NUM}"
TEST_DIR="${FOLD_DIR}/test"
CHECKPOINT_DIR="./checkpoints/cv5_fold_${FOLD_NUM}"
EVAL_RESULTS_DIR="./evaluation_results/cv5_fold_${FOLD_NUM}"

# Check if test data exists
if [ ! -d "$TEST_DIR" ]; then
    echo "Error: Test directory not found: $TEST_DIR"
    echo "Please run preprocessing first:"
    echo "  bash script/preprocess_cv5.sh"
    exit 1
fi

# Create evaluation results directory
mkdir -p "$EVAL_RESULTS_DIR"

echo ""
echo "Configuration:"
echo "  Fold:        ${FOLD_NUM}"
echo "  Test data:   $TEST_DIR"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  Results:     $EVAL_RESULTS_DIR"
echo ""

# Count test samples
TEST_NORMAL=$(find "$TEST_DIR/normal" -name "*.nii.gz" 2>/dev/null | wc -l)
TEST_ALZHEIMER=$(find "$TEST_DIR/alzheimer" -name "*.nii.gz" 2>/dev/null | wc -l)

echo "Test dataset size:"
echo "  Normal:     ${TEST_NORMAL}"
echo "  Alzheimer:  ${TEST_ALZHEIMER}"
echo "  Total:      $((TEST_NORMAL + TEST_ALZHEIMER))"
echo ""

# Default parameters (can be overridden)
MODEL="compact"
TARGET_SHAPE="96 96 96"
BASE_FEATURES=24
SEED=42

# Parse additional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --target_shape)
            TARGET_SHAPE="$2 $3 $4"
            shift 4
            ;;
        --base_features)
            BASE_FEATURES="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run 'python evaluate.py --help' for available options"
            shift
            ;;
    esac
done

# If checkpoint not specified, use default location
if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT="${CHECKPOINT_DIR}/${MODEL}_fold${FOLD_NUM}.pth"
fi

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    echo "Please train the model first:"
    echo "  bash script/train_cv5_fold.sh ${FOLD_NUM}"
    exit 1
fi

echo "Evaluation parameters:"
echo "  Model:        $MODEL"
echo "  Checkpoint:   $CHECKPOINT"
echo "  Target shape: $TARGET_SHAPE"
echo "  Random seed:  $SEED"
echo ""

echo "======================================================================"
echo "Starting evaluation..."
echo "======================================================================"
echo ""

# Run evaluation
python3 evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --data_dir "$TEST_DIR" \
    --model "$MODEL" \
    --target_shape $TARGET_SHAPE \
    --base_features $BASE_FEATURES\
    --seed $SEED \
    --save_dir "$EVAL_RESULTS_DIR"

# Check if evaluation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ Evaluation completed successfully for Fold ${FOLD_NUM}!"
    echo "======================================================================"
    echo ""
    echo "Results saved to: $EVAL_RESULTS_DIR"
    echo ""
    
    # Display results if available
    if [ -f "$EVAL_RESULTS_DIR/metrics.json" ]; then
        echo "Metrics:"
        cat "$EVAL_RESULTS_DIR/metrics.json"
        echo ""
    fi
    
    echo "Next steps:"
    echo "  1. Evaluate other folds: bash script/evaluate_cv5_fold.sh <0-4>"
    echo "  2. Evaluate all folds:   bash script/evaluate_cv5_all.sh"
    echo "  3. Aggregate results:    python aggregate_cv5_results.py"
    echo ""
else
    echo ""
    echo "======================================================================"
    echo "✗ Evaluation failed for Fold ${FOLD_NUM}!"
    echo "======================================================================"
    exit 1
fi
