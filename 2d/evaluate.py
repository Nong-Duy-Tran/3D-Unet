"""
Evaluation script for 2D CNN + Attention Alzheimer's classification
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from model import get_model_2d
from dataset import get_dataloader
from util.utils import plot_confusion_matrix, plot_roc_curve


def set_seed(seed=42):
    """Set random seed for reproducibility"""
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
        for images, labels, _ in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Metrics
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        'accuracy': float(accuracy_score(all_labels, all_preds)),
        'precision': float(precision_score(all_labels, all_preds, zero_division=0)),
        'recall': float(recall_score(all_labels, all_preds, zero_division=0)),
        'specificity': float(recall_score(all_labels, all_preds, pos_label=0, zero_division=0)),
        'f1': float(f1_score(all_labels, all_preds, zero_division=0)),
        'auc': float(roc_auc_score(all_labels, all_probs)) if len(set(all_labels)) > 1 else 0.0,
        'confusion_matrix': cm.tolist(),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'classification_report': classification_report(
            all_labels, all_preds,
            target_names=['Normal', 'Alzheimer'],
            output_dict=True
        ),
    }

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1-Score:    {metrics['f1']:.4f}")
    print(f"AUC-ROC:     {metrics['auc']:.4f}")
    print("=" * 60)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Alzheimer']))

    return metrics, all_labels, all_preds, all_probs


def main(args):
    set_seed(args.seed)
    print(f"Set random seed to: {args.seed}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading data from {args.data_dir} (fold {args.fold})...")
    val_loader = get_dataloader(
        data_dir=args.data_dir,
        fold=args.fold,
        split='test',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Val batches: {len(val_loader)}")

    # Load checkpoint
    print("\nLoading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Build model (prefer args saved in checkpoint)
    ckpt_args = checkpoint.get('args', {})
    model = get_model_2d(
        model_name=ckpt_args.get('model_name', args.model_name),
        in_channels=1,
        num_classes=2,
        num_slices=ckpt_args.get('num_slices', args.num_slices),
        base_channels=ckpt_args.get('base_channels', args.base_channels),
        dropout=ckpt_args.get('dropout', args.dropout),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best Val Acc: {checkpoint.get('best_val_acc', 'unknown')}")

    # Evaluate
    metrics, labels, preds, probs = evaluate_model(model, val_loader, device)

    # Save results
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        print("Generating plots...")
        plot_confusion_matrix(labels, preds, ['Normal', 'Alzheimer'], save_dir / 'confusion_matrix.png')
        plot_roc_curve(labels, probs, save_dir / 'roc_curve.png')

        # Probability distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        normal_probs = probs[labels == 0]
        alzheimer_probs = probs[labels == 1]

        axes[0].hist(normal_probs, bins=20, alpha=0.7, label='Normal', color='blue')
        axes[0].hist(alzheimer_probs, bins=20, alpha=0.7, label='Alzheimer', color='red')
        axes[0].set_xlabel('Probability (Alzheimer class)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Prediction Probability Distribution')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].boxplot([normal_probs, alzheimer_probs], labels=['Normal', 'Alzheimer'])
        axes[1].set_ylabel('Probability (Alzheimer class)')
        axes[1].set_title('Prediction Probability by True Class')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / 'probability_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nResults saved to {args.save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate 2D CNN + Attention model')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data/processed_oasis_2d_cv5',
                        help='Directory with processed 2D data')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold to evaluate (0-4)')
    parser.add_argument('--model_name', type=str, default='standard',
                        choices=['standard', 'compact', 'mrinet'],
                        help='Model architecture')
    parser.add_argument('--num_slices', type=int, default=120,
                        help='Number of slices per volume')
    parser.add_argument('--base_channels', type=int, default=24,
                        help='Base channels for CNN backbone')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()
    main(args)
