#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-3}"
PYTHON_BIN="./.venv/bin/python"
OASIS_DIR="data/OASIS"
NFOLDS="2"
SEED="42"
HDBET_DEVICE="cuda"
LABEL_MODE="cdr4"
TARGET_ROOT="data/oasis_c4"

case "$VERSION" in
  1)
    PREPROCESS_SCRIPT="src/data/v1/preprocess_oasis_data.py"
    POSTPROCESS_SCRIPT="src/data/v1/postprocess_oasis_data.py"
    OUTPUT_DIR_3D="${TARGET_ROOT}/processed_oasis_3d_cv${NFOLDS}_v1"
    OUTPUT_DIR_2D="${TARGET_ROOT}/processed_oasis_2d_cv${NFOLDS}_v1"
    ;;
  2)
    PREPROCESS_SCRIPT="src/data/v2/preprocess_oasis_data.py"
    POSTPROCESS_SCRIPT="src/data/v2/postprocess_oasis_data.py"
    OUTPUT_DIR_3D="${TARGET_ROOT}/processed_oasis_3d_cv${NFOLDS}_v2"
    OUTPUT_DIR_2D="${TARGET_ROOT}/processed_oasis_2d_cv${NFOLDS}_v2"
    ;;
  3)
    PREPROCESS_SCRIPT="src/data/v3/preprocess_oasis_data.py"
    POSTPROCESS_SCRIPT="src/data/v3/postprocess_oasis_data.py"
    OUTPUT_DIR_3D="${TARGET_ROOT}/processed_oasis_3d_cv${NFOLDS}_v3"
    OUTPUT_DIR_2D="${TARGET_ROOT}/processed_oasis_2d_cv${NFOLDS}_v3"
    ;;
  *)
    echo "Invalid pipeline version: $VERSION"
    echo "Usage: $0 [1|2|3]"
    exit 1
    ;;
esac

mkdir -p "${TARGET_ROOT}"

echo "Running OASIS cdr4 pipeline"
echo "  version    : v${VERSION}"
echo "  label mode : ${LABEL_MODE}"
echo "  n_folds    : ${NFOLDS}"
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
