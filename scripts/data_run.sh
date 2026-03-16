#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <1|2|3> [label_mode]"
  echo "  1 -> v1 (nonblank crop)"
  echo "  2 -> v2 (energy crop)"
  echo "  3 -> v3 (largest connected component crop)"
  echo "  label_mode -> cdr4 | normal_vs_nonnormal | ad_vs_nonad | cn_vs_ad_drop_05"
  exit 1
fi

VERSION="$1"
LABEL_MODE="${2:-normal_vs_nonnormal}"
PYTHON_BIN="./.venv/bin/python"
OASIS_DIR="data/OASIS"
NFOLDS="5"
SEED="42"
HDBET_DEVICE="cuda"

case "$VERSION" in
  1)
    PREPROCESS_SCRIPT="src/data/v1/preprocess_oasis_data.py"
    POSTPROCESS_SCRIPT="src/data/v1/postprocess_oasis_data.py"
    OUTPUT_DIR_3D="data/processed_oasis_3d_cv5"
    OUTPUT_DIR_2D="data/processed_oasis_2d_cv5"
    ;;
  2)
    PREPROCESS_SCRIPT="src/data/v2/preprocess_oasis_data.py"
    POSTPROCESS_SCRIPT="src/data/v2/postprocess_oasis_data.py"
    OUTPUT_DIR_3D="data/processed_oasis_3d_cv5_v2"
    OUTPUT_DIR_2D="data/processed_oasis_2d_cv5_v2"
    ;;
  3)
    PREPROCESS_SCRIPT="src/data/v3/preprocess_oasis_data.py"
    POSTPROCESS_SCRIPT="src/data/v3/postprocess_oasis_data.py"
    OUTPUT_DIR_3D="data/processed_oasis_3d_cv5_v3"
    OUTPUT_DIR_2D="data/processed_oasis_2d_cv5_v3"
    ;;
  *)
    echo "Invalid pipeline version: $VERSION"
    echo "Usage: $0 <1|2|3>"
    exit 1
    ;;
esac

echo "Running OASIS pipeline v${VERSION}"
echo "  label mode : ${LABEL_MODE}"
echo "  preprocess : ${PREPROCESS_SCRIPT}"
echo "  postprocess: ${POSTPROCESS_SCRIPT}"
echo "  output 3D  : ${OUTPUT_DIR_3D}"
echo "  output 2D  : ${OUTPUT_DIR_2D}"

"${PYTHON_BIN}" "${PREPROCESS_SCRIPT}" \
  --oasis_dir "${OASIS_DIR}" \
  --output_dir "${OUTPUT_DIR_3D}" \
  --n_folds "${NFOLDS}" \
  --seed "${SEED}" \
  --label_mode "${LABEL_MODE}" \
  --skull_strip \
  --hdbet_device "${HDBET_DEVICE}"

"${PYTHON_BIN}" "${POSTPROCESS_SCRIPT}" \
  --input_dir "${OUTPUT_DIR_3D}" \
  --output_dir "${OUTPUT_DIR_2D}"
