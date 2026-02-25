"""
Training script for Alzheimer's classification baseline
"""
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from monai.losses import FocalLoss
import wandb
from dotenv import load_dotenv

from model import get_model
from dataset import get_dataloaders
from utils import (
    save_checkpoint, load_checkpoint, plot_confusion_matrix,
    plot_roc_curve, plot_training_curves, get_class_weights
)

load_dotenv()


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

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Convert labels to one-hot encoding for FocalLoss
        if isinstance(criterion, FocalLoss):
            labels_onehot = torch.zeros_like(outputs)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            loss = criterion(outputs, labels_onehot)
        else:
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = running_loss / len(dataloader)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    }
    
    return avg_loss, metrics


def validate(model, dataloader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # Convert labels to one-hot encoding for FocalLoss
            if isinstance(criterion, FocalLoss):
                labels_onehot = torch.zeros_like(outputs)
                labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
                loss = criterion(outputs, labels_onehot)
            else:
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = running_loss / len(dataloader)
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    }
    
    predictions = {
        'labels': all_labels,
        'preds': all_preds,
        'probs': all_probs
    }
    
    return avg_loss, metrics, predictions


def main(args):
    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"Set random seed to: {args.seed}")
    
    # Setup
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb_config = {
            'model': args.model,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'loss_fn': args.loss_fn,
            'use_class_weights': args.use_class_weights,
            'target_shape': args.target_shape,
            'base_features': args.base_features,
            'feature_size': args.feature_size,
        }
        
        wandb.init(
            project="Alzheimer Classification MRI",
            name=args.model_name,
            config=wandb_config,
            tags=[args.model, args.loss_fn],
            notes=f"Training {args.model} model for Alzheimer's classification"
        )
    
    # TensorBoard
    writer = SummaryWriter(args.log_dir)
    
    # Data
    print("\nLoading data...")
    train_loader, val_loader = get_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_shape=tuple(args.target_shape),
        use_hdf5=args.use_hdf5
    )
    
    # Model
    print("\nInitializing model...")
    model_kwargs = {
        'in_channels': 1,
        'num_classes': 2,
    }
    
    # Add model-specific parameters
    if args.model == 'swinunet':
        # SwinUNet uses different parameters
        model_kwargs['img_size'] = tuple(args.target_shape)
        model_kwargs['feature_size'] = args.feature_size
        model_kwargs['depths'] = (2, 2, 2, 2)
        model_kwargs['num_heads'] = (3, 6, 12, 24)
        model_kwargs['window_size'] = 7
    else:
        # CNN models use base_features
        model_kwargs['base_features'] = args.base_features
    
    model = get_model(model_name=args.model, **model_kwargs)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss function
    if args.loss_fn == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=1.2, reduction='mean')
        print(f"Using FocalLoss (alpha=0.25, gamma=1.2)")
    elif args.loss_fn == 'bce':
        if args.use_class_weights and hasattr(train_loader.dataset, 'labels'):
            class_weights = get_class_weights(train_loader.dataset.labels)
            pos_weight = torch.FloatTensor([class_weights[1]]).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            print(f"Using BCEWithLogitsLoss with pos_weight: {pos_weight}")
        else:
            criterion = nn.BCEWithLogitsLoss()
            print("Using BCEWithLogitsLoss")
    else:  # crossentropy
        if args.use_class_weights and hasattr(train_loader.dataset, 'labels'):
            class_weights = get_class_weights(train_loader.dataset.labels)
            class_weights = torch.FloatTensor(class_weights).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Using CrossEntropyLoss with class weights: {class_weights}")
        else:
            criterion = nn.CrossEntropyLoss()
            print("Using CrossEntropyLoss")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 100], gamma=0.5
    )
    
    # Training loop
    print("\nStarting training...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    best_val_acc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_metrics, val_preds = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update scheduler
        scheduler.step()
        
        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['auc']:.4f}")
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
        writer.add_scalar('F1/val', val_metrics['f1'], epoch)
        writer.add_scalar('AUC/val', val_metrics['auc'], epoch)
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/accuracy': train_metrics['accuracy'],
                'train/precision': train_metrics['precision'],
                'train/recall': train_metrics['recall'],
                'train/f1': train_metrics['f1'],
                'train/auc': train_metrics['auc'],
                'val/loss': val_loss,
                'val/accuracy': val_metrics['accuracy'],
                'val/precision': val_metrics['precision'],
                'val/recall': val_metrics['recall'],
                'val/f1': val_metrics['f1'],
                'val/auc': val_metrics['auc'],
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch)
        
        # Save history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_metrics['accuracy'])
        val_accs.append(val_metrics['accuracy'])
        
        # Save best model
        if val_accs[-1] >= best_val_acc:
            best_val_acc = val_accs[-1]
            checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.model_name}.pth")
            save_checkpoint(model, optimizer, epoch, best_val_acc, checkpoint_path)
            
            # Save confusion matrix and ROC curve for best model
            cm_path = os.path.join(args.log_dir, 'confusion_matrix.png')
            roc_path = os.path.join(args.log_dir, 'roc_curve.png')
            
            plot_confusion_matrix(
                val_preds['labels'], val_preds['preds'],
                save_path=cm_path
            )
            plot_roc_curve(
                val_preds['labels'], val_preds['probs'],
                save_path=roc_path
            )
            
            # Log to wandb
            if args.use_wandb:
                wandb.log({
                    'best_val_accuracy': best_val_acc,
                    'confusion_matrix': wandb.Image(cm_path),
                    'roc_curve': wandb.Image(roc_path)
                })
                # Save model artifact
                artifact = wandb.Artifact(f'{args.model_name}_model', type='model')
                artifact.add_file(checkpoint_path)
                wandb.log_artifact(artifact)
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Plot training curves
    curves_path = os.path.join(args.log_dir, 'training_curves.png')
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        save_path=curves_path
    )
    
    # Log final training curves to wandb
    if args.use_wandb:
        wandb.log({'training_curves': wandb.Image(curves_path)})
        wandb.finish()
    
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Alzheimer classification baseline')
    
    # Data
    parser.add_argument('--train_dir', type=str, default='../data/train',
                       help='Training data directory')
    parser.add_argument('--val_dir', type=str, default='../data/val',
                       help='Validation data directory')
    parser.add_argument('--use_hdf5', action='store_true',
                       help='Use HDF5 dataset format')
    
    # Model
    parser.add_argument('--model', type=str, default='simple',
                       choices=['simple', 'compact', 'unet', 'resunet', 'swinunet'],
                       help='Model architecture')
    parser.add_argument('--base_features', type=int, default=32,
                       help='Base number of features')
    parser.add_argument('--feature_size', type=int, default=48,
                        help='Feature size for Swin-UNETR')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--use_class_weights', action='store_true',
                       help='Use class weights for imbalanced data')
    parser.add_argument('--loss_fn', type=str, default='crossentropy',
                       choices=['crossentropy', 'bce', 'focal'],
                       help='Loss function to use')
    
    # Data processing
    parser.add_argument('--target_shape', type=int, nargs=3, default=[96, 96, 96],
                       help='Target MRI shape (D H W)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    # Output
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--model_name', type=str, default='best_model',
                       help='Model name for saving')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Log directory')
    
    # Wandb
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    main(args)
