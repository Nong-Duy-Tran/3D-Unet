"""
Evaluation script for trained model
"""
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

from src.baseline.models import get_model
from src.baseline.data import get_dataloaders
from src.utils.plots import plot_confusion_matrix, plot_roc_curve
from src.utils.checkpoint import load_checkpoint


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
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    num_classes = len(set(all_labels))
    if num_classes > 2:
        auc = roc_auc_score(all_labels, np.array(all_probs), multi_class="ovr", average="macro")
        precision = precision_score(all_labels, all_preds, zero_division=0, average="macro")
        recall = recall_score(all_labels, all_preds, zero_division=0, average="macro")
        f1 = f1_score(all_labels, all_preds, zero_division=0, average="macro")
    else:
        auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
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
    if args.class_names:
        print(classification_report(all_labels, all_preds, target_names=args.class_names))
    else:
        print(classification_report(all_labels, all_preds))
    
    return metrics, all_labels, all_preds, all_probs


def _model_kwargs(args):
    kwargs = {
        "in_channels": 1,
        "num_classes": 2,
    }
    if args.model == "simple":
        kwargs["base_features"] = args.base_features
    elif args.model == "vit2d":
        kwargs.update(
            {
                "image_size": args.image_size,
                "patch_size": args.patch_size,
                "embed_dim": args.embed_dim,
                "depth": args.depth,
                "num_heads": args.num_heads,
                "mlp_dim": args.mlp_dim,
                "dropout": args.dropout,
            }
        )
    elif args.model == "swin2d":
        kwargs.update(
            {
                "image_size": args.image_size,
                "patch_size": args.patch_size,
                "embed_dim": args.embed_dim,
                "depth": args.depth,
                "num_heads": args.num_heads,
                "window_size": args.window_size,
                "mlp_dim": args.mlp_dim,
                "dropout": args.dropout,
            }
        )
    elif args.model == "swin3d":
        kwargs.update(
            {
                "patch_size": args.patch_size,
                "embed_dim": args.embed_dim,
                "depth": args.depth,
                "num_heads": args.num_heads,
                "window_size": args.window_size,
                "mlp_dim": args.mlp_dim,
                "dropout": args.dropout,
            }
        )
    elif args.model == "deit2d":
        kwargs.update(
            {
                "timm_name": args.timm_name,
                "pretrained": False,
                "image_size": args.image_size,
                "in_channels": 3 if args.rgb_mode else 1,
                "drop_path_rate": args.drop_path_rate,
                "attn_drop_rate": args.attn_drop_rate,
                "dropout": args.dropout,
            }
        )
    elif args.model in {'vgg2d', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'}:
        vgg_name = args.vgg_name if args.model == "vgg2d" else args.model
        kwargs.update(
            {
                "vgg_name": vgg_name,
                "pretrained": False,
                "image_size": args.image_size,
                "in_channels": 3 if args.rgb_mode else 1,
                "dropout": args.dropout,
            }
        )
    return kwargs


def main(args):
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
        use_2d=args.use_2d,
        num_slices=args.num_slices,
        slice_axis=args.slice_axis,
        resize_2d=args.resize_2d,
        slice_strategy_train="uniform",
        slice_strategy_val="uniform",
        imagenet_norm=args.imagenet_norm,
        randaugment=False,
        rgb_mode=args.rgb_mode,
        normalize=not args.disable_normalize,
    )
    
    # Load model
    print("\nLoading model...")
    model = get_model(model_name=args.model, **_model_kwargs(args))
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
                       choices=[
                           'simple', 'unet', 'resunet', 'vit2d', 'swin2d', 'swin3d', 'deit2d',
                           'vgg2d', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19',
                           'vgg19_bn'
                       ],
                       help='Model architecture')
    parser.add_argument('--base_features', type=int, default=32,
                       help='Base number of features')
    parser.add_argument('--use_2d', action='store_true',
                       help='Use 2D slices instead of 3D volumes')
    parser.add_argument('--num_slices', type=int, default=8,
                       help='Number of slices per volume (2D mode)')
    parser.add_argument('--slice_axis', type=int, default=0,
                       help='Slice axis (0, 1, 2)')
    parser.add_argument('--resize_2d', type=int, nargs=2, default=None,
                       help='Resize 2D slices to (H W)')
    parser.add_argument('--imagenet_norm', action='store_true',
                       help='Apply ImageNet normalization (2D)')
    parser.add_argument('--rgb_mode', action='store_true',
                       help='Convert grayscale to 3-channel (2D)')
    parser.add_argument('--disable_normalize', action='store_true',
                       help='Disable z-score normalization for 3D volumes')
    parser.add_argument('--image_size', type=int, default=64,
                       help='Input image size for ViT')
    parser.add_argument('--patch_size', type=int, default=8,
                       help='Patch size for ViT')
    parser.add_argument('--timm_name', type=str, default='deit_base_patch16_224.fb_in1k',
                       help='timm model name for DeiT')
    parser.add_argument('--vgg_name', type=str, default='vgg16_bn',
                       help='VGG model name for VGG2D')
    parser.add_argument('--drop_path_rate', type=float, default=0.2,
                       help='DeiT drop path rate')
    parser.add_argument('--attn_drop_rate', type=float, default=0.1,
                       help='DeiT attention drop rate')
    parser.add_argument('--window_size', type=int, default=4,
                       help='Window size for Swin')
    parser.add_argument('--embed_dim', type=int, default=256,
                       help='Embedding dim for ViT')
    parser.add_argument('--depth', type=int, default=6,
                       help='ViT depth (num layers)')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='ViT num heads')
    parser.add_argument('--mlp_dim', type=int, default=512,
                       help='ViT MLP dim')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='ViT dropout')
    parser.add_argument('--target_shape', type=int, nargs=3, default=[64, 64, 64],
                       help='Target MRI shape (D H W)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--class_names', type=str, nargs='*', default=None,
                       help='Optional class names for report')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    main(args)
