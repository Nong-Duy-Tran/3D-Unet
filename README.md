# 3D MRI-Based Alzheimer's Disease Classification

A deep learning pipeline for classifying Alzheimer's Disease (AD) from 3D MRI scans using 3D U-Net architecture.

## 📋 Overview

This project implements a complete pipeline for binary classification of Alzheimer's Disease vs Cognitive Normal (CN) subjects using the ADNI dataset. The pipeline includes:

- **Data preprocessing** with subject-based splitting to prevent data leakage
- **3D U-Net models** specifically designed for volumetric MRI classification
- **Complete training pipeline** with TensorBoard monitoring
- **Comprehensive evaluation** with metrics and visualizations

## 🎯 Key Features

- ✅ **True 3D CNNs:** Uses 3D U-Net architecture for volumetric MRI data
- ✅ **nibabel Integration:** Proper NIfTI file handling
- ✅ **Subject-based Splitting:** Prevents data leakage in longitudinal studies
- ✅ **Data Augmentation:** 3D flips, rotations, and intensity scaling
- ✅ **Multiple Models:** SimpleUNet, UNet3D, ResidualUNet3D
- ✅ **Complete Metrics:** Accuracy, Precision, Recall, F1, AUC-ROC
- ✅ **Visualization:** Confusion matrices, ROC curves, training curves

## 📁 Dataset

**ADNI Dataset Structure:**
- **ABB folder:** Alzheimer's Disease (AD) - 500 files from 166 subjects
- **ECC folder:** Cognitive Normal (CN) - 500 files from 208 subjects
- **Total:** 1000 3D MRI scans from 374 unique subjects

**Labels:**
- Class 0: Normal/CN (Cognitively Normal)
- Class 1: Alzheimer's/AD (Alzheimer's Disease)

## 🚀 Quick Start

### Option 1: Automated Pipeline

Run the complete pipeline with a single command:

```bash
./run_pipeline.sh
```

This will:
1. Preprocess and split the data (train/val/test)
2. Train the model
3. Evaluate on the test set

### Option 2: Manual Steps

#### Step 1: Preprocess Data

```bash
# Check statistics first (dry run)
python preprocess_adni_data.py \
    --data_dir ./data/datasets/muhammadzahraan/3d-mri-scans-for-alzheimer-disease/versions/4 \
    --output_dir ./data/processed \
    --dry_run

# Actually process and copy files
python preprocess_adni_data.py \
    --data_dir ./data/datasets/muhammadzahraan/3d-mri-scans-for-alzheimer-disease/versions/4 \
    --output_dir ./data/processed \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42
```

#### Step 2: Train Model

```bash
python train.py \
    --train_dir ./data/processed/train \
    --val_dir ./data/processed/val \
    --model simple \
    --base_features 32 \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4 \
    --target_shape 64 64 64 \
    --use_class_weights \
    --checkpoint_dir ./checkpoints \
    --log_dir ./logs
```

#### Step 3: Monitor Training

```bash
tensorboard --logdir ./logs
```

#### Step 4: Evaluate Model

```bash
python evaluate.py \
    --checkpoint ./checkpoints/best_model.pth \
    --data_dir ./data/processed/test \
    --model simple \
    --base_features 32 \
    --batch_size 4 \
    --target_shape 64 64 64 \
    --save_dir ./evaluation_results
```

## 🏗️ Architecture

### Model: SimpleUNet3DClassifier

```
Input: (batch, 1, 64, 64, 64)
    ↓
[Encoder Block 1] → Conv3D(1→32) + BN + ReLU → Conv3D(32→32) + BN + ReLU
    ↓ MaxPool3D(2)
[Encoder Block 2] → Conv3D(32→64) + BN + ReLU → Conv3D(64→64) + BN + ReLU
    ↓ MaxPool3D(2)
[Encoder Block 3] → Conv3D(64→128) + BN + ReLU → Conv3D(128→128) + BN + ReLU
    ↓ MaxPool3D(2)
[Encoder Block 4] → Conv3D(128→256) + BN + ReLU → Conv3D(256→256) + BN + ReLU
    ↓ MaxPool3D(2)
[Bottleneck] → Conv3D(256→512) + BN + ReLU → Conv3D(512→512) + BN + ReLU
    ↓ AdaptiveAvgPool3D(1)
[Classifier] → Flatten → Dropout(0.5) → Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→2)
    ↓
Output: (batch, 2) - Class logits
```

## 📦 Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory
- 16GB+ RAM

### Install Dependencies

```bash
# Create conda environment (recommended)
conda create -n alzheimer_mri python=3.9
conda activate alzheimer_mri

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

## 📊 Expected Performance

### Baseline Results (SimpleUNet3DClassifier)

- **Accuracy:** 75-85%
- **AUC-ROC:** 0.80-0.90
- **Training Time:** ~10-20 min/epoch (depends on GPU)

### Hardware Requirements

**Minimum:**
- GPU: 8GB VRAM (batch_size=4, 64³ resolution)
- RAM: 16GB
- Storage: ~50GB

**Recommended:**
- GPU: 16GB+ VRAM
- RAM: 32GB+
- Storage: 100GB+

## 📂 Project Structure

```
MRI/
├── preprocess_adni_data.py    # Data preprocessing and splitting
├── dataset.py                  # PyTorch dataset classes
├── model.py                    # 3D U-Net models
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
├── utils.py                    # Helper functions
├── run_pipeline.sh            # Automated pipeline
├── ANALYSIS_REPORT.md         # Comprehensive analysis
├── requirements.txt           # Dependencies
└── data/
    ├── datasets/              # Raw ADNI data
    │   └── muhammadzahraan/
    │       └── 3d-mri-scans-for-alzheimer-disease/
    │           └── versions/4/
    │               ├── abb/   # Alzheimer's Disease
    │               └── ecc/   # Cognitive Normal
    └── processed/             # Processed splits
        ├── train/
        ├── val/
        ├── test/
        └── split_info/
```

## 🔧 Configuration

### Model Options

- `--model simple`: SimpleUNet3DClassifier (recommended for starting)
- `--model unet`: UNet3DClassifier (requires pytorch-3dunet)
- `--model resunet`: ResidualUNet3D (most powerful)

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--target_shape` | 64 64 64 | MRI resolution (D H W) |
| `--batch_size` | 4 | Batch size |
| `--base_features` | 32 | Base number of features |
| `--lr` | 1e-4 | Learning rate |
| `--epochs` | 100 | Number of epochs |
| `--use_class_weights` | False | Balance classes |

### Memory Optimization

If you encounter OOM errors:

1. Reduce batch size: `--batch_size 2` or `--batch_size 1`
2. Reduce resolution: `--target_shape 48 48 48`
3. Reduce features: `--base_features 16`
4. Reduce workers: `--num_workers 2`

## 📈 Monitoring

### TensorBoard

```bash
tensorboard --logdir ./logs
```

**Metrics tracked:**
- Loss (train/val)
- Accuracy (train/val)
- F1 Score
- AUC-ROC
- Precision & Recall

### Outputs

- **Checkpoints:** `./checkpoints/best_model.pth`
- **Training curves:** `./logs/training_curves.png`
- **Confusion matrix:** `./logs/confusion_matrix.png`
- **ROC curve:** `./logs/roc_curve.png`

## 🔍 Data Preprocessing Details

### Subject-Based Splitting ⚠️ Important

The preprocessing script uses **subject-based splitting** to prevent data leakage:

- All scans from the same subject stay in the same split
- This is critical for longitudinal studies with multiple scans per subject
- Ensures the model doesn't learn subject-specific features

### Processing Steps

1. **Scan directories:** Recursively find all NIfTI files
2. **Extract metadata:** Subject IDs, labels, file paths
3. **Split by subject:** 70% train, 15% val, 15% test
4. **Copy files:** Organize into class directories
5. **Save metadata:** JSON and CSV files with split information

## 📚 File Formats

### Input: NIfTI Files

- **Format:** `.nii` (NIfTI-1)
- **Dimensions:** 3D volumetric data
- **Loader:** `nibabel.load()`
- **Normalization:** Z-score (zero mean, unit variance)

### Output: Predictions

- **Format:** Logits (raw scores before softmax)
- **Classes:** [0 (Normal), 1 (Alzheimer's)]
- **Shape:** `(batch_size, 2)`

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory:**
   - Reduce `--batch_size`
   - Reduce `--target_shape`
   - Reduce `--base_features`

2. **Slow Training:**
   - Check GPU utilization: `nvidia-smi`
   - Reduce `--num_workers` if CPU bottleneck
   - Use smaller resolution

3. **Poor Performance:**
   - Train for more epochs
   - Increase model capacity
   - Check data normalization
   - Verify splits are correct

4. **FileNotFoundError:**
   - Check data paths in scripts
   - Ensure preprocessing completed successfully
   - Verify directory structure

## 📞 Support

For detailed analysis and technical specifications, see:
- [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md) - Comprehensive dataset and code analysis

---

**Last Updated:** January 28, 2026
