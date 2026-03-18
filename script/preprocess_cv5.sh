#!/bin/bash

#############################################################################
# OASIS 5-Fold Cross-Validation Data Preprocessing Script
#############################################################################
# This script preprocesses the OASIS dataset for 5-fold cross-validation
# - Creates 5 independent train/test splits (80/20 each)
# - Stratified sampling maintains class balance
# - Subject-level splitting prevents data leakage
# - Fixed seed ensures reproducibility
#############################################################################

echo "======================================================================"
echo "OASIS 5-Fold Cross-Validation Data Preprocessing"
echo "======================================================================"

# Configuration
OASIS_DIR="./data/OASIS"
OUTPUT_DIR="./data/processed_oasis_3d_cv5_v2"
N_FOLDS=5
SEED=42
SKULL_STRIP=true
HDBET_DEVICE="cuda"
LABEL_MODE="normal_vs_nonnormal"
TEAM_ORIENTATION=true

# Parse command line arguments
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
        --n_folds)
            N_FOLDS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --skull_strip)
            SKULL_STRIP=true
            shift
            ;;
        --hdbet_device)
            HDBET_DEVICE="$2"
            shift 2
            ;;
        --label_mode)
            LABEL_MODE="$2"
            shift 2
            ;;
        --team_orientation)
            TEAM_ORIENTATION=true
            shift
            ;;
        --dry_run)
            DRY_RUN="--dry_run"
            shift
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -h|--help)
            echo "Usage: bash script/preprocess_cv5.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --oasis_dir DIR     OASIS data directory (default: ./data/OASIS)"
            echo "  --output_dir DIR    Output directory (default: ./data/processed_oasis_3d_cv5_v2)"
            echo "  --n_folds N         Number of CV folds (default: 5)"
            echo "  --seed SEED         Random seed (default: 42)"
            echo "  --skull_strip       Apply HD-BET skull stripping (RECOMMENDED)"
            echo "  --hdbet_device DEV  HD-BET device: cpu or cuda (default: cuda)"
            echo "  --label_mode MODE   Label mapping mode (default: normal_vs_nonnormal)"
            echo "  --team_orientation  Apply team orientation remap (default: enabled)"
            echo "  --dry_run           Don't copy files, just show statistics"
            echo "  --verbose           Print detailed information"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check if OASIS directory exists
if [ ! -d "$OASIS_DIR" ]; then
    echo "Error: OASIS directory not found: $OASIS_DIR"
    echo "Please download the OASIS dataset first."
    exit 1
fi

# Display configuration
echo ""
echo "Configuration:"
echo "  OASIS directory:  $OASIS_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Number of folds:  $N_FOLDS"
echo "  Random seed:      $SEED"
echo "  Label mode:       $LABEL_MODE"
echo "  Team orientation: $TEAM_ORIENTATION"
echo "  Skull stripping:  $SKULL_STRIP (device=$HDBET_DEVICE)"
echo ""

# Check Python dependencies
echo "Checking Python dependencies..."
python3 -c "import nibabel, sklearn, pandas, numpy, tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Missing required Python packages"
    echo "Please install: nibabel, scikit-learn, pandas, numpy, tqdm"
    echo ""
    echo "Run: pip install nibabel scikit-learn pandas numpy tqdm"
    exit 1
fi
echo "✓ All dependencies found"
echo ""

# Run preprocessing
echo "======================================================================"
echo "Starting preprocessing..."
echo "======================================================================"
echo ""

python process_data/preprocess_oasis_3d.py \
    --oasis_dir "$OASIS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --n_folds $N_FOLDS \
    --seed $SEED \
    --label_mode "$LABEL_MODE" \
    $([ "$SKULL_STRIP" = true ] && echo "--skull_strip --hdbet_device $HDBET_DEVICE") \
    $([ "$TEAM_ORIENTATION" = true ] && echo "--team_orientation") \
    $DRY_RUN \
    $VERBOSE

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ Preprocessing completed successfully!"
    echo "======================================================================"
    echo ""
    echo "Output saved to: $OUTPUT_DIR"
    echo ""
    echo "Directory structure:"
    echo "  $OUTPUT_DIR/"
    echo "    ├── _cache/  (<session_id>.nii.gz)"
    echo "    └── split_info/  (metadata)"
    echo ""
    echo "Next steps:"
    echo "  1. Export 2D slices from cache + split_info"
    echo "  2. Train/evaluate by fold metadata in split_info/folds.json"
    echo ""
else
    echo ""
    echo "======================================================================"
    echo "✗ Preprocessing failed!"
    echo "======================================================================"
    exit 1
fi
