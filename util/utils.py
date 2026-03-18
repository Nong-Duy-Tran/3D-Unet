"""
Utility functions for training and evaluation
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


def save_checkpoint(model, optimizer, epoch, best_val_acc, filepath,
                    best_val_loss=None, best_val_auc=None):
    """Save model checkpoint.

    The historical argument name `best_val_acc` is preserved for backward
    compatibility with existing callers. Additional metrics can be optionally
    stored when available.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc
    }

    if best_val_loss is not None:
        checkpoint['best_val_loss'] = best_val_loss
    if best_val_auc is not None:
        checkpoint['best_val_auc'] = best_val_auc
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath, device='cuda'):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint.get('best_val_loss', None)
    best_val_acc = checkpoint.get('best_val_acc', None)
    
    print(f"Checkpoint loaded from {filepath}")
    if best_val_loss is not None:
        print(f"Epoch: {epoch}, Best val loss: {best_val_loss:.4f}")
        return epoch, best_val_loss

    if best_val_acc is not None:
        print(f"Epoch: {epoch}, Best val acc: {best_val_acc:.4f}")
        return epoch, best_val_acc

    print(f"Epoch: {epoch}")
    return epoch, None


def plot_confusion_matrix(y_true, y_pred, classes=['Normal', 'Alzheimer'], save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path=None):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    plt.close()


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    plt.close()


def get_class_weights(labels):
    """Calculate class weights for imbalanced datasets"""
    labels = np.array(labels)
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    
    class_weights = total_samples / (len(class_counts) * class_counts)
    
    return class_weights
