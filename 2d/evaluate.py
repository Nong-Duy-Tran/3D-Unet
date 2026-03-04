"""
Evaluation script for 2D CNN + Attention Alzheimer's classification
"""
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from model import get_model_2d
from dataset import get_dataloader
from util.utils import load_checkpoint, plot_confusion_matrix, plot_roc_curve


def evaluate_model(model, dataloader, device, save_predictions=False):
    """
    Evaluate model on given dataloader
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to run evaluation on
        save_predictions: Whether to save detailed predictions
    
    Returns:
        metrics: Dictionary of evaluation metrics
        predictions: Dictionary of predictions (if save_predictions=True)
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_subject_ids = []
    
    print("\nEvaluating...")
    with torch.no_grad():
        for images, labels, subject_ids in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of Alzheimer's class
            all_subject_ids.extend(subject_ids)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(all_labels, all_preds)),
        'precision': float(precision_score(all_labels, all_preds, zero_division=0)),
        'recall': float(recall_score(all_labels, all_preds, zero_division=0)),
        'f1': float(f1_score(all_labels, all_preds, zero_division=0)),
        'specificity': float(recall_score(all_labels, all_preds, pos_label=0, zero_division=0)),
        'auc': float(roc_auc_score(all_labels, all_probs)) if len(set(all_labels)) > 1 else 0.0
    }
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Per-class metrics
    tn, fp, fn, tp = cm.ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # Classification report
    class_names = ['Normal', 'Alzheimer']
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    metrics['classification_report'] = report
    
    # Predictions
    predictions = None
    if save_predictions:
        predictions = {
            'subject_ids': all_subject_ids,
            'labels': all_labels.tolist(),
            'predictions': all_preds.tolist(),
            'probabilities': all_probs.tolist()
        }
    
    return metrics, predictions, all_labels, all_preds, all_probs


def evaluate_fold(args, fold):
    """Evaluate a single fold"""
    print(f"\n{'='*70}")
    print(f"Evaluating Fold {fold}")
    print(f"{'='*70}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    fold_name = f'fold_{fold}'
    eval_dir = Path(args.eval_dir) / fold_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    val_loader = get_dataloader(
        data_dir=args.data_dir,
        fold=fold,
        split='val',
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Val batches: {len(val_loader)}")
    
    # Load model
    print("\nLoading model...")
    checkpoint_path = Path(args.checkpoint_dir) / fold_name / 'best_model.pth'
    
    if not checkpoint_path.exists():
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model args from checkpoint if available
    if 'args' in checkpoint:
        model_args = checkpoint['args']
        model = get_model_2d(
            model_name=model_args.get('model_name', args.model_name),
            in_channels=1,
            num_classes=2,
            num_slices=model_args.get('num_slices', args.num_slices),
            base_channels=model_args.get('base_channels', args.base_channels),
            dropout=model_args.get('dropout', args.dropout)
        )
    else:
        model = get_model_2d(
            model_name=args.model_name,
            in_channels=1,
            num_classes=2,
            num_slices=args.num_slices,
            base_channels=args.base_channels,
            dropout=args.dropout
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best Val Acc: {checkpoint.get('best_val_acc', 'unknown')}")
    print(f"Best Val AUC: {checkpoint.get('best_val_auc', 'unknown')}")
    
    # Evaluate
    metrics, predictions, labels, preds, probs = evaluate_model(
        model, val_loader, device, save_predictions=args.save_predictions
    )
    
    # Print metrics
    print("\n" + "="*70)
    print("Evaluation Metrics")
    print("="*70)
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1-Score:    {metrics['f1']:.4f}")
    print(f"AUC:         {metrics['auc']:.4f}")
    print("\nConfusion Matrix:")
    print(f"  TN: {metrics['true_negatives']}, FP: {metrics['false_positives']}")
    print(f"  FN: {metrics['false_negatives']}, TP: {metrics['true_positives']}")
    
    # Save metrics
    with open(eval_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions if requested
    if args.save_predictions and predictions is not None:
        with open(eval_dir / 'predictions.json', 'w') as f:
            json.dump(predictions, f, indent=2)
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Confusion matrix
    plot_confusion_matrix(
        labels, preds, 
        ['Normal', 'Alzheimer'],
        eval_dir / 'confusion_matrix.png'
    )
    
    # ROC curve
    plot_roc_curve(labels, probs, eval_dir / 'roc_curve.png')
    
    # Additional plots
    # Distribution of predictions by class
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Probability distribution for each class
    normal_probs = probs[labels == 0]
    alzheimer_probs = probs[labels == 1]
    
    axes[0].hist(normal_probs, bins=20, alpha=0.7, label='Normal', color='blue')
    axes[0].hist(alzheimer_probs, bins=20, alpha=0.7, label='Alzheimer', color='red')
    axes[0].set_xlabel('Probability (Alzheimer class)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Prediction Probability Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    data_to_plot = [normal_probs, alzheimer_probs]
    axes[1].boxplot(data_to_plot, labels=['Normal', 'Alzheimer'])
    axes[1].set_ylabel('Probability (Alzheimer class)')
    axes[1].set_title('Prediction Probability by True Class')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(eval_dir / 'probability_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults saved to: {eval_dir}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate 2D CNN + Attention model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./data/processed_oasis_2d_cv5',
                       help='Directory with processed 2D data')
    parser.add_argument('--fold', type=int, default=0,
                       help='Fold number to evaluate (0-4), or -1 for all folds')
    parser.add_argument('--num_slices', type=int, default=120,
                       help='Number of slices per volume')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='compact', choices=['standard', 'compact'],
                       help='Model architecture')
    parser.add_argument('--base_channels', type=int, default=24,
                       help='Base channels for CNN backbone (24 for compact, 32 for standard)')

    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save detailed predictions')
    
    # Directories
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/2d_cv5',
                       help='Directory with saved checkpoints')
    parser.add_argument('--eval_dir', type=str, default='./evaluation_results/2d_cv5',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 70)
    print("2D CNN + Attention Alzheimer's Classification Evaluation")
    print("=" * 70)
    print("\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    # Evaluate
    if args.fold == -1:
        # Evaluate all folds
        print("\n" + "=" * 70)
        print("Evaluating all 5 folds")
        print("=" * 70)
        
        all_metrics = []
        for fold in range(5):
            metrics = evaluate_fold(args, fold)
            if metrics is not None:
                all_metrics.append({
                    'fold': fold,
                    **{k: v for k, v in metrics.items() 
                       if k not in ['confusion_matrix', 'classification_report']}
                })
        
        if len(all_metrics) > 0:
            # Compute average metrics
            print("\n" + "=" * 70)
            print("5-Fold Cross-Validation Results")
            print("=" * 70)
            
            metric_names = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc']
            summary = {}
            
            for metric_name in metric_names:
                values = [m[metric_name] for m in all_metrics if metric_name in m]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    summary[f'{metric_name}_mean'] = float(mean_val)
                    summary[f'{metric_name}_std'] = float(std_val)
                    print(f"{metric_name.capitalize():12s}: {mean_val:.4f} ± {std_val:.4f}")
            
            # Save summary
            summary['fold_results'] = all_metrics
            summary_path = Path(args.eval_dir) / 'cv5_summary.json'
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\nSummary saved to: {summary_path}")
        else:
            print("\nNo folds were successfully evaluated.")
    
    else:
        # Evaluate single fold
        evaluate_fold(args, args.fold)


if __name__ == '__main__':
    main()
