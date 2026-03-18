"""
Training script for 2D CNN + Attention Alzheimer's classification
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path
import json
import wandb
from dotenv import load_dotenv

from model import get_model_2d
from dataset import get_dataloader
from util.utils import (
    save_checkpoint, load_checkpoint, plot_confusion_matrix,
    plot_roc_curve, plot_training_curves, get_class_weights
)

load_dotenv()


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


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, use_amp=False):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler(device='cuda') if use_amp else None
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, labels, _) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if use_amp:
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
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
    all_subject_ids = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for images, labels, subject_ids in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())
            all_subject_ids.extend(subject_ids)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = running_loss / len(dataloader)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    }
    
    return avg_loss, metrics, all_labels, all_preds, all_probs


def main(args, fold):
    """Train a single fold"""
    print(f"\n{'='*70}")
    print(f"Training Fold {fold}")
    print(f"{'='*70}")
    
    # Set seed for this fold
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    fold_name = f'fold_{fold}'
    checkpoint_dir = Path(args.checkpoint_dir) / fold_name
    log_dir = Path(args.log_dir) / fold_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir)
    
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.exp_name}_fold{fold}",
            config=vars(args),
            reinit=True
        )
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader = get_dataloader(
        data_dir=args.data_dir,
        fold=fold,
        split='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=None,
        slice_transform=None
    )
    
    val_loader = get_dataloader(
        data_dir=args.data_dir,
        fold=fold,
        split='val',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=None,
        slice_transform=None
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = get_model_2d(
        model_name=args.model_name,
        in_channels=1,
        num_classes=2,
        num_slices=args.num_slices,
        base_channels=args.base_channels,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    if args.use_class_weights:
        # Calculate class weights from training data
        train_labels = train_loader.dataset.labels
        class_weights = get_class_weights(train_labels)
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using weighted CrossEntropyLoss with weights: {class_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using CrossEntropyLoss")
    
    # Optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                            momentum=0.9, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=10)
    else:
        scheduler = None
    
    # Load checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_auc = 0.0
    
    if args.resume:
        checkpoint_path = checkpoint_dir / 'best_model.pth'
        if checkpoint_path.exists():
            print(f"\nResuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            best_val_auc = checkpoint.get('best_val_auc', 0.0)
            if best_val_loss != float('inf'):
                print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
            else:
                print(f"Resumed from epoch {start_epoch}, best acc: {best_val_acc:.4f}")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_auc': [],
        'val_auc': []
    }
    
    # Training loop
    print(f"\n{'='*70}")
    print("Starting training...")
    print(f"{'='*70}")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1, args.use_amp
        )
        
        # Validate
        val_loss, val_metrics, val_labels, val_preds, val_probs = validate(
            model, val_loader, criterion, device, epoch+1
        )
        
        # Update scheduler
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['train_auc'].append(train_metrics['auc'])
        history['val_auc'].append(val_metrics['auc'])
        
        # Print metrics
        print(f"\nTrain - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"AUC: {train_metrics['auc']:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"AUC: {val_metrics['auc']:.4f}")
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
        writer.add_scalar('AUC/train', train_metrics['auc'], epoch)
        writer.add_scalar('AUC/val', val_metrics['auc'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Wandb logging
        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_metrics['accuracy'],
                'val_acc': val_metrics['accuracy'],
                'train_auc': train_metrics['auc'],
                'val_auc': val_metrics['auc'],
                'val_recall': val_metrics['recall'],
                'val_precision': val_metrics['precision'],
                'val_f1': val_metrics['f1'],
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Save checkpoint (use fold-specific directory)
        best_checkpoint_path = checkpoint_dir / 'best_model.pth'
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_val_acc = val_metrics['accuracy']
            best_val_auc = val_metrics['auc']
            
            # Use utils.py save_checkpoint function
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_val_acc,
                best_checkpoint_path,
                best_val_loss=best_val_loss,
                best_val_auc=best_val_auc
            )
            print(f"\u2713 Saved best model (Val Loss: {best_val_loss:.4f})")
        
        # Save periodic checkpoint every N epochs
        if (epoch + 1) % args.save_freq == 0:
            periodic_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics['accuracy'],
                periodic_path,
                best_val_loss=val_loss,
                best_val_auc=val_metrics['auc']
            )
            print(f"Saved checkpoint at epoch {epoch+1}")
    
    # Save final plots
    print("\nGenerating plots...")
    plot_training_curves(history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'], log_dir / 'training_curves.png')
    plot_confusion_matrix(val_labels, val_preds, ['Normal', 'Alzheimer'], 
                         log_dir / 'confusion_matrix.png')
    plot_roc_curve(val_labels, val_probs, log_dir / 'roc_curve.png')
    
    # Save final metrics
    final_metrics = {
        'fold': fold,
        'best_val_loss': float(best_val_loss),
        'best_val_acc': float(best_val_acc),
        'best_val_auc': float(best_val_auc),
        'final_train_loss': float(history['train_loss'][-1]),
        'final_val_loss': float(history['val_loss'][-1]),
        'final_train_acc': float(history['train_acc'][-1]),
        'final_val_acc': float(history['val_acc'][-1])
    }
    
    with open(checkpoint_dir / 'metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
	
    metadata = {
        "base_channel": args.base_channels,
        "model_name": args.model_name,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "scheduler": args.scheduler,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "use_class_weights": args.use_class_weights,
        "use_amp": args.use_amp,
        "seed": args.seed
	}
    
    with open(checkpoint_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)

    writer.close()
    
    if args.use_wandb:
        wandb.finish()
    
    print(f"\n{'='*70}")
    print(f"Fold {fold} training completed!")
    print(f"Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.4f}, Best Val AUC: {best_val_auc:.4f}")
    print(f"{'='*70}")
    
    return best_val_acc, best_val_auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train 2D CNN + Attention model for Alzheimer classification')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./data/processed_oasis_2d_cv5',
                       help='Directory with processed 2D data')
    parser.add_argument('--fold', type=int, default=0,
                       help='Fold number to train (0-4), or -1 for all folds')
    parser.add_argument('--num_slices', type=int, default=120,
                       help='Number of slices per volume')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='compact',
                       choices=['standard', 'compact', 'mrinet', 'thresholded', 'gaussian_init'],
                       help='Model architecture')
    parser.add_argument('--base_channels', type=int, default=24,
                       help='Base channels for CNN backbone (24 for compact, 32 for standard)')

    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='plateau', 
                       choices=['cosine', 'step', 'plateau', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=30,
                       help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='Gamma for StepLR scheduler')
    parser.add_argument('--use_class_weights', action='store_true',
                       help='Use class weights for imbalanced data')
    parser.add_argument('--use_amp', action='store_true',
                       help='Use automatic mixed precision training')
    
    # Directories
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/2d_cv5',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs/2d_cv5',
                       help='Directory to save logs')
    
    # Other parameters
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_freq', type=int, default=20,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    
    # Wandb parameters
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='alzheimer-2d-classification',
                       help='Wandb project name')
    parser.add_argument('--exp_name', type=str, default='2d_cnn_attention',
                       help='Experiment name')
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 70)
    print("2D CNN + Attention Alzheimer's Classification Training")
    print("=" * 70)
    print("\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    # Train
    if args.fold == -1:
        # Train all folds
        print("\n" + "=" * 70)
        print("Training all 5 folds")
        print("=" * 70)
        
        fold_results = []
        for fold in range(5):
            best_acc, best_auc = main(args, fold)
            fold_results.append({
                'fold': fold,
                'best_val_acc': best_acc,
                'best_val_auc': best_auc
            })
        
        # Print summary
        print("\n" + "=" * 70)
        print("5-Fold Cross-Validation Results")
        print("=" * 70)
        for result in fold_results:
            print(f"Fold {result['fold']}: Acc={result['best_val_acc']:.4f}, AUC={result['best_val_auc']:.4f}")
        
        avg_acc = np.mean([r['best_val_acc'] for r in fold_results])
        avg_auc = np.mean([r['best_val_auc'] for r in fold_results])
        std_acc = np.std([r['best_val_acc'] for r in fold_results])
        std_auc = np.std([r['best_val_auc'] for r in fold_results])
        
        print(f"\nAverage: Acc={avg_acc:.4f}±{std_acc:.4f}, AUC={avg_auc:.4f}±{std_auc:.4f}")
        
        # Save summary
        summary = {
            'fold_results': fold_results,
            'average_acc': float(avg_acc),
            'std_acc': float(std_acc),
            'average_auc': float(avg_auc),
            'std_auc': float(std_auc)
        }
        
        summary_dir = Path(args.checkpoint_dir)
        summary_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_dir / 'cv5_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
    else:
        # Train single fold
        main(args, args.fold)

