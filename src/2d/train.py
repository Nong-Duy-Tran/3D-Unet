"""
Training script for 2D CNN + Attention Alzheimer's classification
Uses Backbone + MultiAttention (models.py) + OASIS2DDataset (dataset.py)
"""
import os
import sys
import random
import json
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ColorJitter, Compose, Resize, ToTensor, Normalize, RandomAffine
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# Optional dependencies
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from models import MyDenseNetMultiAttention, MyResNetMultiAttention, MyMobileNetMultiAttention
from dataset import OASIS2DDataset


# ---------------------------------------------------------------------------
# Utility helpers (no external util module required)
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, metric, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metric': metric,
    }, path)


def load_checkpoint(path, model, optimizer=None, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt.get('epoch', 0), ckpt.get('metric', 0.0)


def get_class_weights(labels):
    classes, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    weights = total / (len(classes) * counts)
    return weights.astype(np.float32)


def plot_training_curves(train_loss, val_loss, train_acc, val_acc, save_path):
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_loss, label='Train'); axes[0].plot(val_loss, label='Val')
    axes[0].set_title('Loss'); axes[0].legend()
    axes[1].plot(train_acc, label='Train'); axes[1].plot(val_acc, label='Val')
    axes[1].set_title('Accuracy'); axes[1].legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_confusion_matrix(labels, preds, class_names, save_path):
    if not HAS_MPL:
        return
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_roc_curve(labels, probs, save_path):
    if not HAS_MPL:
        return
    from sklearn.metrics import roc_curve, auc
    
    # Convert to numpy arrays and filter out NaN values
    labels = np.array(labels)
    probs = np.array(probs)
    valid_mask = ~np.isnan(probs)
    labels = labels[valid_mask]
    probs = probs[valid_mask]
    
    # Skip if no valid data
    if len(probs) == 0:
        print("Warning: No valid probabilities for ROC curve (all NaN)")
        return
    
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC Curve')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_metrics_curves(history, save_path):
    """
    Plot per-epoch precision, recall, and AUC for both train and val splits.

    Args:
        history  : dict with keys 'train_precision', 'val_precision',
                   'train_recall', 'val_recall', 'train_auc', 'val_auc'.
        save_path: Path-like – output PNG path.
    """
    if not HAS_MPL:
        return
    epochs = range(1, len(history['train_precision']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ['precision', 'recall', 'auc']):
        ax.plot(epochs, history[f'train_{metric}'], label='Train')
        ax.plot(epochs, history[f'val_{metric}'], label='Val')
        ax.set_title(metric.capitalize())
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, scaler=None):
    """Train for one epoch"""
    model.train()
    use_amp = scaler is not None
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, labels, _) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if use_amp:
            with torch.amp.autocast(device_type='cuda'):
                outputs, _ = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Check for NaN loss and skip batch if detected
        loss_item = loss.item()
        if np.isnan(loss_item) or np.isinf(loss_item):
            print(f"\nWARNING: NaN/Inf loss detected at batch {batch_idx}. Skipping batch update.")
            continue
        
        running_loss += loss_item

        # Check for NaN in outputs before computing metrics
        if torch.isnan(outputs).any():
            print(f"\nWARNING: NaN values in model outputs at batch {batch_idx}. Skipping metrics computation.")
            continue
            
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss_item:.4f}'})

    avg_loss = running_loss / len(dataloader)
    
    # Calculate metrics
    try:
        auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
        auc = auc if not np.isnan(auc) else 0.0
    except (ValueError, RuntimeWarning):
        auc = 0.0
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': auc
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
            
            outputs, _ = model(images)
            loss = criterion(outputs, labels)

            # Check for NaN loss
            loss_item = loss.item()
            if np.isnan(loss_item) or np.isinf(loss_item):
                print("\nWARNING: NaN/Inf loss in validation. Skipping batch.")
                continue

            running_loss += loss_item

            # Check for NaN in outputs
            if torch.isnan(outputs).any():
                print("\nWARNING: NaN values in validation outputs. Skipping batch.")
                continue
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())
            all_subject_ids.extend(subject_ids)
            
            pbar.set_postfix({'loss': f'{loss_item:.4f}'})

    avg_loss = running_loss / max(len(dataloader), 1)
    
    # Filter out NaN values from probabilities
    valid_indices = [i for i, p in enumerate(all_probs) if not np.isnan(p)]
    all_probs = [all_probs[i] for i in valid_indices]
    all_labels = [all_labels[i] for i in valid_indices]
    all_preds = [all_preds[i] for i in valid_indices]
    
    if len(all_probs) == 0 or len(all_labels) == 0:
        print("Warning: All validation predictions contain NaN values or no valid samples")
        return avg_loss, {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc': 0.0
        }, [], [], []
    
    # Calculate metrics
    try:
        auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
        auc = auc if not np.isnan(auc) else 0.0
    except (ValueError, RuntimeWarning):
        auc = 0.0
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': auc
    }
    
    return avg_loss, metrics, all_labels, all_preds, all_probs


def main(args, fold):
    """Train a single fold"""
    # Auto-compute num_slices from view unless explicitly overridden
    view_slice_map = {'all': 240, 'axial': 80, 'coronal': 80, 'sagittal': 80}
    if args.num_slices is None:
        args.num_slices = view_slice_map[args.view]

    print(f"\n{'='*70}")
    print(f"Training Fold {fold} | View: {args.view.upper()} ({args.num_slices} slices)")
    print(f"{'='*70}")

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Output directories
    fold_name = f'fold_{fold}'
    checkpoint_dir = Path(args.checkpoint_dir) / fold_name
    log_dir = Path(args.log_dir) / fold_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Remove stale checkpoint files from previous runs before starting fresh
    for file in checkpoint_dir.glob('*.pth'):
        try:
            file.unlink()
            print(f'[INFO] Removed old checkpoint: {file}')
        except Exception as e:
            print(f'[WARNING] Could not remove file {file}: {e}')
    if log_dir.exists():
        import shutil
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    writer = None
    if HAS_TENSORBOARD:
        writer = SummaryWriter(log_dir)

    # Wandb
    if args.use_wandb:
        if not HAS_WANDB:
            print("[Warning] wandb not installed – skipping wandb logging.")
            args.use_wandb = False
        else:
            wandb.init(
                project=args.wandb_project,
                name=f"{args.exp_name}_fold{fold}",
                config=vars(args),
                reinit=True
            )

    train_transform = Compose([
        RandomAffine(degrees=(-5, 5), scale=(0.9, 1.1)),
    ])
    
    val_transform = Compose([])  

    # Datasets & DataLoaders
    print("\nLoading data...")
    train_dataset = OASIS2DDataset(
        data_dir=args.data_dir,
        fold=fold,
        split='train',
        view=args.view,
        one_transform=None,
    )
    val_dataset = OASIS2DDataset(
        data_dir=args.data_dir,
        fold=fold,
        split='val',
        view=args.view,
        one_transform=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val   batches: {len(val_loader)}")

    # Model
    print("\nCreating model....")
    model = MyMobileNetMultiAttention(
        num_classes=2,
        num_slices=args.num_slices,
        embed_dim=args.embed_dim,
    )
    model = model.to(device)

    # ------------------------------------------------------------------
    # Freeze backbone (optional)
    # ------------------------------------------------------------------
    if args.freeze_backbone:
        if not args.trainable_layers:
            # Empty list → unfreeze entire backbone (end-to-end training through backbone)
            print("\n[INFO] trainable_layers is empty → Backbone fully unfrozen.")
            for param in model.backbone.parameters():
                param.requires_grad = True
        else:
            # Non-empty list → freeze all backbone, then selectively unfreeze matching layers
            print("\n[INFO] Freezing CNN Backbone...")
            for param in model.backbone.parameters():
                param.requires_grad = False
            print(f"[INFO] Unfreezing layers matching: {args.trainable_layers}")
            for name, param in model.backbone.named_parameters():
                if any(layer in name for layer in args.trainable_layers):
                    param.requires_grad = True
                    print(f"  [unfrozen] {name}")
    else:
        print("\n[INFO] Backbone is NOT frozen (end-to-end training). Pass --no-freeze-backbone to disable freezing.")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss
    if args.use_class_weights:
        class_weights = get_class_weights(train_dataset.labels)
        alpha = 1.0
        class_weights[1] = class_weights[1] * alpha
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
        print(f"Using weighted CrossEntropyLoss: {class_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print("Using CrossEntropyLoss")
    
    # Optimizer — only parameters with requires_grad=True
    trainable_params_list = list(filter(lambda p: p.requires_grad, model.parameters()))
    if args.optimizer == 'adam':
        optimizer = optim.Adam(trainable_params_list, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(trainable_params_list, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(trainable_params_list, lr=args.lr,
                              momentum=0.9, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.3, patience=10)
    else:
        scheduler = None
    
    # Mixed-precision scaler
    scaler = torch.amp.GradScaler('cuda') if args.use_amp and device.type == 'cuda' else None

    # Resume from checkpoint
    start_epoch = 0
    best_val_acc = 0.0
    best_val_auc = 0.0
    best_loss = float('inf')
    early_stop_counter = 0
    best_score = 0.0

    if args.resume:
        checkpoint_path = checkpoint_dir / 'best_model.pth'
        if checkpoint_path.exists():
            print(f"\nResuming from checkpoint: {checkpoint_path}")
            start_epoch, best_loss = load_checkpoint(
                checkpoint_path, model, optimizer, device=device
            )
            print(f"Resumed from epoch {start_epoch}, best loss: {best_loss:.4f}")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_auc': [],
        'val_auc': [],
        'train_precision': [],
        'val_precision': [],
        'train_recall': [],
        'val_recall': [],
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
            model, train_loader, criterion, optimizer, device, epoch+1, scaler
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
        history['train_precision'].append(train_metrics['precision'])
        history['val_precision'].append(val_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['val_recall'].append(val_metrics['recall'])
        
        # Print metrics
        print(f"\nTrain - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"AUC: {train_metrics['auc']:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"Recall: {val_metrics['recall']:.4f}, Precision: {val_metrics['precision']:.4f}, "
              f"AUC: {val_metrics['auc']:.4f}")
        
        # TensorBoard logging
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
            writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            writer.add_scalar('AUC/train', train_metrics['auc'], epoch)
            writer.add_scalar('AUC/val', val_metrics['auc'], epoch)
            writer.add_scalar('Precision/train', train_metrics['precision'], epoch)
            writer.add_scalar('Precision/val', val_metrics['precision'], epoch)
            writer.add_scalar('Recall/train', train_metrics['recall'], epoch)
            writer.add_scalar('Recall/val', val_metrics['recall'], epoch)
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
                'train_precision': train_metrics['precision'],
                'val_precision': val_metrics['precision'],
                'train_recall': train_metrics['recall'],
                'val_recall': val_metrics['recall'],
                'lr': optimizer.param_groups[0]['lr'],
            })
        
        # Save best checkpoint
        best_checkpoint_path = checkpoint_dir / 'best_model.pth'

        # Ensure AUC and F1 are valid (not NaN)
        val_auc = val_metrics['auc'] if not np.isnan(val_metrics['auc']) else 0.0
        val_f1 = val_metrics['f1'] if not np.isnan(val_metrics['f1']) else 0.0
        val_recall = val_metrics['recall'] if not np.isnan(val_metrics['recall']) else 0.0
        val_precision = val_metrics['precision'] if not np.isnan(val_metrics['precision']) else 0.0

        current_score = (val_recall * 0.2) + (val_auc * 0.4) + (val_f1 * 0.4)
        if current_score > best_score:
            best_loss = val_loss
            best_val_acc = val_metrics['accuracy']
            best_val_auc = val_auc  # Use already-validated AUC value
            best_score = current_score
            early_stop_counter = 0
            save_checkpoint(model, optimizer, epoch, best_loss, best_checkpoint_path)
            print(f"\u2713 Saved best model (Val Loss: {best_loss:.4f} | Val AUC: {best_val_auc:.4f} | Val Acc: {best_val_acc:.4f})")
        else:
            early_stop_counter += 1
            print(f"  Early stopping: {early_stop_counter}/{args.early_stopping_patience} epochs without improvement")
            if args.early_stopping_patience > 0 and early_stop_counter >= args.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs (patience={args.early_stopping_patience}).")
                break

        # Periodic checkpoint
        if (epoch + 1) % args.save_freq == 0:
            periodic_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint(model, optimizer, epoch, best_loss, periodic_path)
            print(f"Saved checkpoint at epoch {epoch+1}")
    
    # Save final plots
    print("\nGenerating plots...")
    plot_training_curves(history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'], log_dir / 'training_curves.png')
    plot_metrics_curves(history, log_dir / 'metrics_curves.png')
    plot_confusion_matrix(val_labels, val_preds, ['Normal', 'Alzheimer'], 
                         log_dir / 'confusion_matrix.png')
    plot_roc_curve(val_labels, val_probs, log_dir / 'roc_curve.png')
    
    # Save final metrics
    final_metrics = {
        'fold': fold,
        'best_val_acc': float(best_val_acc),
        'best_val_auc': float(best_val_auc) if not np.isnan(best_val_auc) else 0.0,
        'final_train_loss': float(history['train_loss'][-1]),
        'final_val_loss': float(history['val_loss'][-1]),
        'final_train_acc': float(history['train_acc'][-1]),
        'final_val_acc': float(history['val_acc'][-1])
    }
    
    with open(checkpoint_dir / 'metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    if writer is not None:
        writer.close()

    if args.use_wandb and HAS_WANDB:
        wandb.finish()
    
    print(f"\n{'='*70}")
    print(f"Fold {fold} training completed!")
    print(f"Best Val Acc: {best_val_acc:.4f}, Best Val AUC: {best_val_auc:.4f}")
    print(f"{'='*70}")
    
    return best_val_acc, best_val_auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Baseline for Alzheimer classification')

    # Data
    parser.add_argument('--data_dir', type=str, default='./data/processed_oasis_2d_jpeg',
                       help='Root directory containing axial/, coronal/, sagittal/ JPEG folders')
    parser.add_argument('--fold', type=int, default=0,
                       help='Fold to train (0-4), or -1 for all folds')
    parser.add_argument('--view', type=str, default='sagittal',
                       choices=['all', 'axial', 'coronal', 'sagittal'],
                       help='MRI view to use: all (80), axial (16), coronal (32), sagittal (32)')
    parser.add_argument('--num_slices', type=int, default=None,
                       help='Override number of slices (auto-computed from --view if not set)')

    # Model
    parser.add_argument('--embed_dim', type=int, default=128,
                       help='Embedding dimension for the attention layer')
    parser.add_argument('--freeze_backbone', default=True, action=argparse.BooleanOptionalAction,
                       help='Freeze CNN backbone (default: True). Pass --no-freeze-backbone for end-to-end training.')
    parser.add_argument('--trainable_layers', type=str, nargs='*',
                       default=[],
                       help='Layer name patterns inside backbone to unfreeze when --freeze_backbone is set '
                            '(e.g. --trainable_layers denseblock4 norm5)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate (reduced default to 5e-4 for stability)')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='plateau', 
                       choices=['cosine', 'step', 'plateau', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=40,
                       help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='Gamma for StepLR scheduler')
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                       help='Use class weights for imbalanced data')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='Label smoothing factor for CrossEntropyLoss')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use automatic mixed precision training')
    
    # Directories
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/2d_cv5',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs/2d_cv5',
                       help='Directory to save logs')
    
    # Other parameters
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=50,
                       help='Stop training if val loss does not improve for this many epochs (0 = disabled)')
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
        
        # Clean NaN values before computing averages
        val_accs = [r['best_val_acc'] if not np.isnan(r['best_val_acc']) else 0.0 for r in fold_results]
        val_aucs = [r['best_val_auc'] if not np.isnan(r['best_val_auc']) else 0.0 for r in fold_results]
        
        avg_acc = np.mean(val_accs)
        avg_auc = np.mean(val_aucs)
        std_acc = np.std(val_accs)
        std_auc = np.std(val_aucs)
        
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
