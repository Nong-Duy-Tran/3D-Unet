# ADNI Alzheimer's Disease Dataset - Comprehensive Analysis Report

**Date:** January 28, 2026  
**Purpose:** Analysis and preprocessing for 3D MRI-based Alzheimer's classification

---

## 1. Dataset Analysis

### 1.1 Dataset Structure

**Location:** `/home/ntq/Projects/longtd/MRI/data/datasets/muhammadzahraan/3d-mri-scans-for-alzheimer-disease/versions/4/`

**Directory Organization:**
```
versions/4/
├── abb/ADNI/          # Alzheimer's Disease (AD) - Label 1
│   ├── 166 subjects
│   └── 500 NIfTI files (.nii)
├── ecc/ADNI/          # Cognitive Normal (CN) - Label 0
│   ├── 208 subjects
│   └── 500 NIfTI files (.nii)
└── bbc/               # (Not used per requirement)
```

### 1.2 File Structure Pattern

Each NIfTI file follows this nested structure:
```
[label]/ADNI/[SUBJECT_ID]/[PROCESSING_TYPE]/[SCAN_DATE]/[IMAGE_ID]/[filename].nii

Example:
abb/ADNI/002_S_0619/MPR-R__GradWarp__N3__Scaled/2006-06-01_20_04_45.0/I48617/
    ADNI_002_S_0619_MR_MPR-R__GradWarp__N3__Scaled_Br_20070411125458928_S15145_I48617.nii
```

### 1.3 Dataset Statistics

| Category | Subjects | Files | Label |
|----------|----------|-------|-------|
| **ABB (Alzheimer's)** | 166 | 500 | 1 |
| **ECC (Normal)** | 208 | 500 | 0 |
| **Total** | 374 | 1000 | - |

**Key Observations:**
- Balanced dataset: 500 AD vs 500 CN files
- Multiple scans per subject (longitudinal data)
- Various preprocessing pipelines (MPR, GradWarp, N3, B1 Correction, etc.)

### 1.4 CSV Metadata Files

- `abb_9_13_2025.csv`: 7,654 rows - AD scan metadata
- `ecc_9_20_2025.csv`: 14,295 rows - CN scan metadata

**Metadata includes:**
- Image Data ID
- Subject ID
- Group (AD/CN)
- Demographics (Sex, Age)
- Visit information
- Acquisition date
- Processing description

---

## 2. Current Code Analysis

### 2.1 Model Architecture ✅

**File:** [model.py](model.py)

**Available Models:**

1. **SimpleUNet3DClassifier** (Recommended)
   - Custom 3D U-Net encoder for classification
   - 5 encoder blocks with progressive downsampling
   - Global average pooling + classifier head
   - **Architecture:** ✅ True 3D CNN for volumetric MRI
   - **Input:** (batch, 1, D, H, W) - single channel 3D MRI
   - **Output:** (batch, 2) - binary classification logits
   - **Base features:** Configurable (default: 32)

2. **UNet3DClassifier**
   - Uses `pytorch-3dunet` library's U-Net encoder
   - More sophisticated with residual connections
   - Global pooling + classification head

3. **ResidualUNet3DClassifier**
   - Similar to UNet3DClassifier with residual blocks

**Conclusion:** ✅ **Models are properly designed as 3D U-Net architectures for AD/CN classification**

### 2.2 Dataset Handler ✅

**File:** [dataset.py](dataset.py)

**Key Features:**
- ✅ Uses `nibabel` for loading NIfTI files
- ✅ Handles 3D MRI volumes properly
- ✅ Supports data augmentation (flips, rotations, intensity scaling)
- ✅ Z-score normalization
- ✅ Automatic resizing to target shape
- ✅ Expected directory structure: `root/normal/` and `root/alzheimer/`
- ✅ Also supports HDF5 format

**Data Augmentation:**
- Random 3D flips (along all axes)
- Random 3D rotations (-10° to +10°)
- Random intensity scaling (0.9x to 1.1x)

### 2.3 Training Pipeline ✅

**File:** [train.py](train.py)

**Features:**
- ✅ Full training loop with validation
- ✅ TensorBoard logging
- ✅ Model checkpointing
- ✅ Learning rate scheduling (ReduceLROnPlateau)
- ✅ Class weight support for imbalanced data
- ✅ Comprehensive metrics (accuracy, precision, recall, F1, AUC)

### 2.4 Evaluation ✅

**File:** [evaluate.py](evaluate.py)

**Features:**
- ✅ Complete evaluation pipeline
- ✅ Confusion matrix visualization
- ✅ ROC curve plotting
- ✅ Detailed classification report

---

## 3. Required Changes & Implementation

### 3.1 Data Preprocessing ✅ COMPLETED

**Created:** [preprocess_adni_data.py](preprocess_adni_data.py)

**Features:**
- ✅ Scans both `abb` (AD) and `ecc` (CN) folders
- ✅ Recursively finds all NIfTI files
- ✅ **Subject-based splitting** to avoid data leakage
- ✅ Creates train/val/test splits (70%/15%/15% default)
- ✅ Organizes files into expected structure
- ✅ Saves split metadata (JSON & CSV)
- ✅ Generates summary statistics
- ✅ Dry-run mode for testing

**Output Structure:**
```
data/processed/
├── train/
│   ├── normal/
│   └── alzheimer/
├── val/
│   ├── normal/
│   └── alzheimer/
├── test/
│   ├── normal/
│   └── alzheimer/
└── split_info/
    ├── train_split.json
    ├── val_split.json
    ├── test_split.json
    ├── train_split.csv
    ├── val_split.csv
    ├── test_split.csv
    └── summary.json
```

### 3.2 Existing Code Status

**No major changes needed!** The existing code is already well-designed for 3D MRI classification:

1. ✅ **dataset.py** - Already uses nibabel and handles 3D volumes correctly
2. ✅ **model.py** - Already implements 3D U-Net for classification
3. ✅ **train.py** - Complete training pipeline ready to use
4. ✅ **evaluate.py** - Complete evaluation pipeline
5. ✅ **utils.py** - Helper functions for checkpoints, plotting, etc.

---

## 4. Step-by-Step Usage Guide

### Step 1: Preprocess the Dataset

Run the preprocessing script to create train/val/test splits:

```bash
# Dry run first to check statistics
python preprocess_adni_data.py \
    --data_dir ./data/datasets/muhammadzahraan/3d-mri-scans-for-alzheimer-disease/versions/4 \
    --output_dir ./data/processed \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42 \
    --dry_run

# Actually copy files
python preprocess_adni_data.py \
    --data_dir ./data/datasets/muhammadzahraan/3d-mri-scans-for-alzheimer-disease/versions/4 \
    --output_dir ./data/processed \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42
```

### Step 2: Train the Model

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

**Recommended Parameters:**
- `--model simple`: Start with SimpleUNet3DClassifier
- `--base_features 32`: Balance between capacity and memory
- `--batch_size 4`: Adjust based on GPU memory (reduce if OOM)
- `--target_shape 64 64 64`: Smaller size for faster training, increase for better accuracy

### Step 3: Evaluate the Model

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

### Step 4: Monitor Training

```bash
tensorboard --logdir ./logs
```

---

## 5. Technical Specifications

### 5.1 Input Data Format

- **File Format:** NIfTI (.nii)
- **Data Type:** 3D volumetric MRI
- **Input Shape:** Variable (will be resized to target_shape)
- **Channels:** 1 (grayscale)
- **Normalization:** Z-score normalization (zero mean, unit variance)

### 5.2 Model Input/Output

- **Input Shape:** `(batch_size, 1, D, H, W)` where D, H, W = target_shape
- **Output Shape:** `(batch_size, 2)` - logits for 2 classes
- **Classes:** 0 (Normal/CN), 1 (Alzheimer's/AD)

### 5.3 Hardware Requirements

**Minimum:**
- GPU: 8GB VRAM (for batch_size=4, target_shape=64x64x64)
- RAM: 16GB
- Storage: ~50GB for processed data

**Recommended:**
- GPU: 16GB+ VRAM (for larger batch sizes or higher resolution)
- RAM: 32GB+
- Storage: 100GB+

### 5.4 Dependencies

Core libraries (see [requirements.txt](requirements.txt)):
- `torch` - PyTorch framework
- `nibabel` - NIfTI file loading ✅
- `numpy` - Numerical operations
- `scipy` - Image processing (zoom, rotate)
- `sklearn` - Metrics and evaluation
- `matplotlib` - Visualization
- `seaborn` - Statistical plotting
- `h5py` - HDF5 support (optional)
- `tensorboard` - Training monitoring
- `tqdm` - Progress bars

---

## 6. Data Splitting Strategy

### 6.1 Subject-Based Splitting ✅

**Critical:** The preprocessing script uses **subject-based splitting** to prevent data leakage:

- All scans from the same subject are kept in the same split
- Prevents the model from learning subject-specific features
- Ensures proper generalization evaluation

### 6.2 Split Ratios

- **Train:** 70% of subjects (~262 subjects, ~700 files)
- **Validation:** 15% of subjects (~56 subjects, ~150 files)
- **Test:** 15% of subjects (~56 subjects, ~150 files)

---

## 7. Expected Results

### 7.1 Baseline Performance

With the SimpleUNet3DClassifier, you can expect:

- **Accuracy:** 75-85% (typical for AD vs CN classification)
- **AUC:** 0.80-0.90
- **Training Time:** ~10-20 minutes per epoch (depends on hardware)

### 7.2 Improvement Strategies

1. **Increase resolution:** Use target_shape=(96, 96, 96) or (128, 128, 128)
2. **Increase model capacity:** Use more base_features (64, 96)
3. **Use transfer learning:** Try UNet3DClassifier with pytorch-3dunet
4. **Ensemble:** Train multiple models with different seeds
5. **Advanced augmentation:** Add elastic deformation, noise injection

---

## 8. Summary & Recommendations

### ✅ Current Status

1. **Dataset:** ✅ Analyzed and understood
   - 1000 files total (500 AD, 500 CN)
   - 374 unique subjects
   - Well-balanced dataset

2. **Model:** ✅ Properly designed for 3D classification
   - True 3D U-Net architecture
   - Suitable for volumetric MRI
   - Multiple model options available

3. **Code:** ✅ Ready to use
   - nibabel for NIfTI loading ✅
   - 3D data augmentation ✅
   - Complete training pipeline ✅

4. **Preprocessing:** ✅ Implemented
   - New script for data splitting
   - Subject-based splitting to avoid leakage
   - Proper directory structure creation

### 🚀 Next Steps

1. **Run preprocessing:**
   ```bash
   python preprocess_adni_data.py --data_dir ./data/datasets/muhammadzahraan/3d-mri-scans-for-alzheimer-disease/versions/4 --output_dir ./data/processed
   ```

2. **Start training:**
   ```bash
   python train.py --train_dir ./data/processed/train --val_dir ./data/processed/val
   ```

3. **Monitor and iterate:**
   - Watch TensorBoard for training progress
   - Adjust hyperparameters as needed
   - Evaluate on test set after training

### 📊 Key Insights

1. **Data Quality:** The ADNI dataset is high-quality with standardized preprocessing
2. **Balance:** Dataset is well-balanced (no need for heavy class weighting)
3. **Subjects:** Multiple scans per subject - perfect for subject-based splitting
4. **Architecture:** 3D U-Net is appropriate for this task
5. **Ready to Go:** All components are in place for training!

---

## 9. Troubleshooting

### Common Issues

1. **Out of Memory (OOM):**
   - Reduce `--batch_size` (try 2 or 1)
   - Reduce `--target_shape` (try 48 48 48)
   - Reduce `--base_features` (try 16)

2. **Slow Training:**
   - Reduce `--target_shape`
   - Reduce `--num_workers`
   - Use smaller model

3. **Poor Performance:**
   - Increase model capacity
   - Increase target resolution
   - Train for more epochs
   - Check data normalization

---

**End of Analysis Report**
