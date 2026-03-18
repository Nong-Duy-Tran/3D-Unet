#!/bin/bash

PYTHON="C:/Users/Lenovo/miniconda3/envs/alzheimer-baseline/python.exe"
TRAIN="d:/YEAR 3/Lab/3D-Unet/src/2d/train.py"

for FOLD in 0 1 2 3 4; do
    echo "========== Training fold $FOLD =========="
    "$PYTHON" "$TRAIN" --fold $FOLD
    echo "========== Fold $FOLD done =========="
done

echo "All folds completed."
