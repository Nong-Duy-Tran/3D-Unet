#!/bin/bash
# Preprocessing script for 2D OASIS dataset with 5-fold CV
# Input: pre-processed skull-stripped data (processed_oasis_cv5_skullstrip)
# Output: .npz files with (img_size, img_size, num_slices) 2D axial slices
#
# Usage:
#   ./script/preprocess_2d.sh
#   ./script/preprocess_2d.sh --img_size 256 --num_slices 128
#   ./script/preprocess_2d.sh --dry_run --verbose

echo "=========================================="
echo "OASIS 2D Data Preprocessing"
echo "=========================================="

# Default parameters
OASIS_DIR="./data/processed_oasis_cv5_skullstrip"
OUTPUT_DIR="./data/processed_oasis_2d_cv5"
IMG_SIZE=224
NUM_SLICES=120
NORM_METHOD="percentile"
RAS_ORIENT=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --oasis_dir)
            OASIS_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --img_size)
            IMG_SIZE="$2"
            shift 2
            ;;
        --num_slices)
            NUM_SLICES="$2"
            shift 2
            ;;
        --no_ras_orient)
            RAS_ORIENT=false
            shift
            ;;
        --norm_method)
            NORM_METHOD="$2"
            shift 2
            ;;
        --dry_run)
            DRY_RUN="--dry_run"
            shift
            ;;
        --verbose)
            VERBOSE="--verbose"
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
echo "  Input directory:  $OASIS_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Image size:       ${IMG_SIZE}x${IMG_SIZE}"
echo "  Number of slices: $NUM_SLICES"
echo "  RAS orientation:  $RAS_ORIENT"
echo "  Skull stripping:  PRE-APPLIED"
echo "  Normalization:    $NORM_METHOD"
echo ""

# Build command
CMD="python process_data/preprocess_oasis_2d.py \
    --oasis_dir $OASIS_DIR \
    --output_dir $OUTPUT_DIR \
    --img_size $IMG_SIZE \
    --num_slices $NUM_SLICES \
    --norm_method $NORM_METHOD"

if [ "$RAS_ORIENT" = true ]; then
    CMD="$CMD --ras_orient"
fi

if [ ! -z "$DRY_RUN" ]; then
    CMD="$CMD $DRY_RUN"
fi

if [ ! -z "$VERBOSE" ]; then
    CMD="$CMD $VERBOSE"
fi

# Run preprocessing
echo "=========================================="
echo "Starting preprocessing..."
echo "=========================================="
echo ""

eval $CMD

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Preprocessing completed successfully!"
    echo "=========================================="
    echo "Output directory: $OUTPUT_DIR"
else
    echo ""
    echo "=========================================="
    echo "Preprocessing failed with exit code $EXIT_CODE"
    echo "=========================================="
    exit $EXIT_CODE
fi
