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
from dataset import get_dataloader
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
        elif isinstance(criterion, nn.BCEWithLogitsLoss):
            labels_bce = labels.float()
            logits = outputs.squeeze(1) if outputs.ndim == 2 and outputs.size(1) == 1 else outputs
            loss = criterion(logits, labels_bce)
        else:
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            logits = outputs.squeeze(1) if outputs.ndim == 2 and outputs.size(1) == 1 else outputs
            probs_pos = torch.sigmoid(logits)
            preds = (probs_pos >= 0.5).long()
            all_probs.extend(probs_pos.detach().cpu().numpy())
        else:
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].detach().cpu().numpy())
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = running_loss / len(dataloader)
    
    # Calculate specificity
    all_labels_arr = np.array(all_labels)
    all_preds_arr = np.array(all_preds)
    tn = np.sum((all_labels_arr == 0) & (all_preds_arr == 0))
    fp = np.sum((all_labels_arr == 0) & (all_preds_arr == 1))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'specificity': specificity,
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
            elif isinstance(criterion, nn.BCEWithLogitsLoss):
                labels_bce = labels.float()
                logits = outputs.squeeze(1) if outputs.ndim == 2 and outputs.size(1) == 1 else outputs
                loss = criterion(logits, labels_bce)
            else:
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                logits = outputs.squeeze(1) if outputs.ndim == 2 and outputs.size(1) == 1 else outputs
                probs_pos = torch.sigmoid(logits)
                preds = (probs_pos >= 0.5).long()
                all_probs.extend(probs_pos.cpu().numpy())
            else:
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                all_probs.extend(probs[:, 1].cpu().numpy())
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = running_loss / len(dataloader)
    
    # Calculate specificity
    all_labels_arr = np.array(all_labels)
    all_preds_arr = np.array(all_preds)
    tn = np.sum((all_labels_arr == 0) & (all_preds_arr == 0))
    fp = np.sum((all_labels_arr == 0) & (all_preds_arr == 1))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'specificity': specificity,
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
    fold_name = f'fold_{args.fold}'
    checkpoint_dir = os.path.join(args.checkpoint_dir, fold_name)
    log_dir = os.path.join(args.log_dir, fold_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.exp_name}_fold{args.fold}",
            config=vars(args),
            reinit=True,
            group=args.wandb_group,
        )
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    # Data
    print("\nLoading data...")
    train_loader = get_dataloader(
        data_dir=args.data_dir,
        fold=args.fold,
        split='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_shape=tuple(args.target_shape)
    )
    val_loader = get_dataloader(
        data_dir=args.data_dir,
        fold=args.fold,
        split='val',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_shape=tuple(args.target_shape)
    )
    
    # Model
    print("\nInitializing model...")
    model_kwargs = {
        'in_channels': 1,
        'num_classes': 1 if args.loss_fn == 'bce' else 2,
    }
    
    # Add model-specific parameters
    if args.model_name == 'swinunet':
        # SwinUNet uses different parameters
        model_kwargs['img_size'] = tuple(args.target_shape)
        model_kwargs['feature_size'] = args.feature_size
        model_kwargs['depths'] = (2, 2, 2, 2)
        model_kwargs['num_heads'] = (3, 6, 12, 24)
        model_kwargs['window_size'] = 7
    else:
        # CNN models use base_features
        model_kwargs['base_features'] = args.base_features
    
    model = get_model(model_name=args.model_name, **model_kwargs)
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
    
    start_epoch = 1
    best_val_loss = float('inf')
    best_val_acc = 0.0

    # Resume from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"\nResuming from checkpoint: {args.resume}")
            resume_epoch, c_best_val_loss, c_best_val_acc = load_checkpoint(
                model, optimizer, args.resume, device=device, scheduler=scheduler
            )
            start_epoch = resume_epoch + 1
            if c_best_val_loss != float('inf'):
                best_val_loss = c_best_val_loss
            if c_best_val_acc != 0.0:
                best_val_acc = c_best_val_acc
            
            # Fast-forward the scheduler only if it wasn't loaded from the checkpoint (for backward compatibility)
            if scheduler.last_epoch == 0:
                for _ in range(start_epoch - 1):
                    scheduler.step()
        else:
            print(f"\nWarning: Checkpoint '{args.resume}' not found. Starting from scratch.")
    
    # Training loop
    print("\nStarting training...")
    print("=" * 60)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(start_epoch, args.epochs + 1):
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
        print(f"Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f} | Val Specificity: {val_metrics['specificity']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['auc']:.4f}")
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
        writer.add_scalar('Specificity/train', train_metrics['specificity'], epoch)
        writer.add_scalar('Specificity/val', val_metrics['specificity'], epoch)
        writer.add_scalar('F1/val', val_metrics['f1'], epoch)
        writer.add_scalar('AUC/val', val_metrics['auc'], epoch)
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                'train/loss': train_loss,
                'train/accuracy': train_metrics['accuracy'],
                'train/precision': train_metrics['precision'],
                'train/recall': train_metrics['recall'],
                'train/specificity': train_metrics['specificity'],
                'train/f1': train_metrics['f1'],
                'train/auc': train_metrics['auc'],
            }, step=epoch)
            
            wandb.log({
                'val/loss': val_loss,
                'val/accuracy': val_metrics['accuracy'],
                'val/precision': val_metrics['precision'],
                'val/recall': val_metrics['recall'],
                'val/specificity': val_metrics['specificity'],
                'val/f1': val_metrics['f1'],
                'val/auc': val_metrics['auc'],
            }, step=epoch)
        
        # Save history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_metrics['accuracy'])
        val_accs.append(val_metrics['accuracy'])
        
        # Save best model based on lowest validation loss
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_metrics['accuracy']
            checkpoint_path = os.path.join(checkpoint_dir, f"{args.model_name}.pth")
            save_checkpoint(
                model, optimizer, epoch, best_val_acc, checkpoint_path, 
                best_val_loss=best_val_loss, scheduler=scheduler
            )
            
            # Save confusion matrix and ROC curve for best model
            cm_path = os.path.join(log_dir, 'confusion_matrix.png')
            roc_path = os.path.join(log_dir, 'roc_curve.png')
            
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
                'info/epoch': epoch,
                'info/learning_rate': optimizer.param_groups[0]['lr'],
                'info/best_val_loss': best_val_loss,
                'info/best_val_accuracy': best_val_acc,
            }, step=epoch)
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Alzheimer classification baseline')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='../data',
                       help='Data directory containing train and val splits')
    parser.add_argument('--fold', type=int, default=0,
                       help='Fold index for tracking/logging')
    
    # Model
    parser.add_argument('--model_name', type=str, default='simple',
                       choices=['simple', 'compact', 'unet', 'resunet', 'swinunet', 'brainiac', 'vit_unetr'],
                       help='Model architecture')
    parser.add_argument('--base_features', type=int, default=32,
                       help='Base number of features')
    parser.add_argument('--feature_size', type=int, default=48,
                        help='Feature size for Swin-UNETR')
    
    # Training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
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
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Log directory')
    
    # Wandb
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='alzheimer-3d-classification',
                       help='Wandb project name')
    parser.add_argument('--exp_name', type=str, default='3d_model',
                       help='Experiment name')
    parser.add_argument('--wandb_group', type=str, default=None,
                       help='Group name')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    main(args)
