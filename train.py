"""
Training script for Alzheimer's classification baseline
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from model import get_model
from dataset import get_dataloaders
from utils import (
    save_checkpoint, load_checkpoint, plot_confusion_matrix,
    plot_roc_curve, plot_training_curves, get_class_weights
)


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
    # Setup
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    model = get_model(
        model_name=args.model,
        in_channels=1,
        num_classes=2,
        base_features=args.base_features
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss with class weights
    if args.use_class_weights and hasattr(train_loader.dataset, 'labels'):
        class_weights = get_class_weights(train_loader.dataset.labels)
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using class weights: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
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
        scheduler.step(val_loss)
        
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
        
        # Save history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_metrics['accuracy'])
        val_accs.append(val_metrics['accuracy'])
        
        # Save best model
        if val_accs[-1] >= best_val_acc:
            best_val_acc = val_accs[-1]
            save_checkpoint(
                model, optimizer, epoch, best_val_loss, best_val_acc,
                os.path.join(args.checkpoint_dir, 'best_model.pth')
            )
            
            # Save confusion matrix and ROC curve for best model
            plot_confusion_matrix(
                val_preds['labels'], val_preds['preds'],
                save_path=os.path.join(args.log_dir, 'confusion_matrix.png')
            )
            plot_roc_curve(
                val_preds['labels'], val_preds['probs'],
                save_path=os.path.join(args.log_dir, 'roc_curve.png')
            )
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Plot training curves
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        save_path=os.path.join(args.log_dir, 'training_curves.png')
    )
    
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
                       choices=['simple', 'unet', 'resunet'],
                       help='Model architecture')
    parser.add_argument('--base_features', type=int, default=32,
                       help='Base number of features')
    
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
    
    # Data processing
    parser.add_argument('--target_shape', type=int, nargs=3, default=[64, 64, 64],
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
    
    args = parser.parse_args()
    main(args)
