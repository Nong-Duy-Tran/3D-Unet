# Alzheimer's Classification Baseline using pytorch-3dunet

This baseline adapts the pytorch-3dunet library for binary classification of Alzheimer's disease from 3D MRI scans.

## Important Note

**pytorch-3dunet is designed for segmentation, not classification**. We've created a custom wrapper to adapt the U-Net architecture for classification by:
1. Using the U-Net encoder as a feature extractor
2. Adding global average pooling
3. Adding a classification head

## Installation

```bash
# Create conda environment
conda create -n alzheimer-baseline python=3.11 -y
conda activate alzheimer-baseline

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation (OASIS 3D standard flow)

Process raw OASIS discs into 3D NIfTI with the group's standard pipeline:

```bash
python src/data/preprocess_oasis_data.py --oasis_dir data/OASIS --output_dir data/processed_oasis_3d_cv5 --n_folds 5 --val_ratio 0.15 --seed 42 --skull_strip --hdbet_device cuda
```


## Training (Hydra + Lightning)

Single run:
```bash
python -m src.project.cli.train
```

Select a grouped experiment:
```bash
python -m src.project.cli.train experiment=oasis_deit
```

5-fold run:
```bash
python scripts/train_kfold.py --config-name oasis_vgg --folds 5
```

## Evaluation

```bash
python -m src.baseline.evaluate --checkpoint checkpoints/best_model.pth --data_dir data/val
```

## Metrics

The baseline computes:
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC
- Confusion Matrix

## Architecture

- **Encoder**: 3D U-Net encoder (from pytorch-3dunet)
- **Pooling**: Global Average Pooling 3D
- **Classifier**: Fully connected layers with dropout
- **Output**: 2 classes (Normal, Alzheimer)
