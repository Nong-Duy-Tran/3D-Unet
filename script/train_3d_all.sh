#!/bin/bash

#############################################################################
# Train all 5 folds sequentially for 5-Fold Cross-Validation
#############################################################################
# Usage: bash script/train_cv5_all.sh [OPTIONS]
# Example: bash script/train_cv5_all.sh --model unet --epochs 200
#############################################################################

echo "======================================================================"
echo "Training All Folds - 5-Fold Cross-Validation"
echo "======================================================================"
echo ""
echo "This will train all 5 folds sequentially."
echo "Each fold takes significant time (depending on epochs and data size)."
echo ""

# Check if data exists
DATA_ROOT="./data/processed_oasis_cv5"
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Processed data not found: $DATA_ROOT"
    echo "Please run preprocessing first:"
    echo "  bash script/preprocess_cv5.sh"
    exit 1
fi

# Capture all arguments to pass to individual fold training
ARGS="$@"

echo "Training parameters: $ARGS"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Array to track success/failure
declare -a RESULTS

# Train each fold
for FOLD_NUM in {0..4}; do
    echo ""
    echo "======================================================================"
    echo "FOLD ${FOLD_NUM}/5"
    echo "======================================================================"
    echo ""
    
    # Train this fold
    bash script/train_cv5_fold.sh $FOLD_NUM $ARGS
    
    # Check result
    if [ $? -eq 0 ]; then
        RESULTS[$FOLD_NUM]="✓ SUCCESS"
        echo "✓ Fold ${FOLD_NUM} completed successfully"
    else
        RESULTS[$FOLD_NUM]="✗ FAILED"
        echo "✗ Fold ${FOLD_NUM} failed!"
        
        # Ask if user wants to continue
        echo ""
        read -p "Continue with remaining folds? (y/n) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Training stopped by user."
            break
        fi
    fi
    
    echo ""
    echo "Progress:"
    for i in {0..4}; do
        if [ -z "${RESULTS[$i]}" ]; then
            echo "  Fold $i: [PENDING]"
        else
            echo "  Fold $i: ${RESULTS[$i]}"
        fi
    done
    echo ""
done

# Summary
echo ""
echo "======================================================================"
echo "Training Summary - All Folds"
echo "======================================================================"
echo ""

SUCCESS_COUNT=0
FAIL_COUNT=0

for i in {0..4}; do
    if [ -z "${RESULTS[$i]}" ]; then
        echo "  Fold $i: [NOT RUN]"
    elif [[ "${RESULTS[$i]}" == *"SUCCESS"* ]]; then
        echo "  Fold $i: ✓ SUCCESS"
        ((SUCCESS_COUNT++))
    else
        echo "  Fold $i: ✗ FAILED"
        ((FAIL_COUNT++))
    fi
done

echo ""
echo "Total: ${SUCCESS_COUNT} successful, ${FAIL_COUNT} failed"
echo ""

if [ $SUCCESS_COUNT -eq 5 ]; then
    echo "======================================================================"
    echo "✓ All folds trained successfully!"
    echo "======================================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate all folds: bash script/evaluate_cv5_all.sh"
    echo "  2. Aggregate results:  python aggregate_cv5_results.py"
    echo ""
    exit 0
else
    echo "======================================================================"
    echo "⚠ Some folds failed to train"
    echo "======================================================================"
    echo ""
    echo "Please check the error messages above and retry failed folds."
    exit 1
fi
