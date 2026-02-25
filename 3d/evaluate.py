"""
Evaluation script for trained model
"""
import os
import random
import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

from model import get_model
from dataset import get_dataloaders
from util.utils import plot_confusion_matrix, plot_roc_curve, load_checkpoint


def set_seed(seed=42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def evaluate_model(model, dataloader, device):
    """Evaluate model on dataset"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    }
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"AUC-ROC:   {metrics['auc']:.4f}")
    print("=" * 60)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Alzheimer']))
    
    return metrics, all_labels, all_preds, all_probs


def main(args):    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"Set random seed to: {args.seed}")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    _, val_loader = get_dataloaders(
        train_dir=args.data_dir,  # Dummy, won't be used
        val_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_shape=tuple(args.target_shape),
        use_hdf5=args.use_hdf5
    )
    
    # Load model
    print("\nLoading model...")
    model_kwargs = {
        'in_channels': 1,
        'num_classes': 2,
    }
    
    # Add model-specific parameters
    if args.model == 'swinunet':
        # SwinUNet uses different parameters
        model_kwargs['img_size'] = tuple(args.target_shape)
        model_kwargs['feature_size'] = args.feature_size
        model_kwargs['depths'] = tuple(args.depths)
        model_kwargs['num_heads'] = tuple(args.num_heads)
        model_kwargs['window_size'] = args.window_size
    else:
        # CNN models use base_features
        model_kwargs['base_features'] = args.base_features
    
    model = get_model(model_name=args.model, **model_kwargs)
    model = model.to(device)
    
    # Load checkpoint
    load_checkpoint(model, None, args.checkpoint, device)
    
    # Evaluate
    metrics, labels, preds, probs = evaluate_model(model, val_loader, device)
    
    # Save plots
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        
        plot_confusion_matrix(
            labels, preds,
            save_path=os.path.join(args.save_dir, 'confusion_matrix.png')
        )
        
        plot_roc_curve(
            labels, probs,
            save_path=os.path.join(args.save_dir, 'roc_curve.png')
        )
        
        print(f"\nResults saved to {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Alzheimer classification model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to evaluation data directory')
    parser.add_argument('--model', type=str, default='simple',
                       choices=['simple', 'unet', 'resunet', 'swinunet', 'compact'],
                       help='Model architecture')
    parser.add_argument('--base_features', type=int, default=32,
                       help='Base number of features (for CNN models)')
    
    # Swin-UNETR specific parameters
    parser.add_argument('--feature_size', type=int, default=48,
                       help='Feature size for Swin-UNETR')
    parser.add_argument('--depths', type=int, nargs=4, default=[2, 2, 2, 2],
                       help='Depths for Swin-UNETR layers')
    parser.add_argument('--num_heads', type=int, nargs=4, default=[3, 6, 12, 24],
                       help='Number of attention heads for Swin-UNETR')
    parser.add_argument('--window_size', type=int, default=7,
                       help='Window size for Swin-UNETR')
    
    parser.add_argument('--use_hdf5', action='store_true',
                       help='Use HDF5 dataset format')
    parser.add_argument('--target_shape', type=int, nargs=3, default=[96, 96, 96],
                       help='Target MRI shape (D H W)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    main(args)
