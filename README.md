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

## Data Preparation

Convert your MRI data to HDF5 format:

```bash
```


## Training

Lightning + Hydra + W&B:
```bash
python -m src.baseline.train
```

Override examples:
```bash
python -m src.baseline.train training.max_epochs=20 data.batch_size=8
python -m src.baseline.train logging.wandb.enabled=false
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
