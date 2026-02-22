"""
Visualization Utilities
Plotting functions for training curves, sample images, and metrics
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curves(history, output_path=None):
    """
    Plot training and validation loss/accuracy curves
    
    Args:
        history: dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    if 'train_acc' in history:
        axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    if 'val_acc' in history:
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved training curves to: {output_path}")
    
    plt.close()


def plot_gan_losses(g_losses, d_losses, output_path=None):
    """
    Plot GAN training losses
    
    Args:
        g_losses: Generator losses per epoch
        d_losses: Discriminator/Critic losses per epoch
        output_path: Path to save figure
    """
    plt.figure(figsize=(10, 5))
    
    epochs = range(1, len(g_losses) + 1)
    
    plt.plot(epochs, g_losses, 'b-', label='Generator Loss', linewidth=2, alpha=0.8)
    plt.plot(epochs, d_losses, 'r-', label='Discriminator Loss', linewidth=2, alpha=0.8)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved GAN loss curves to: {output_path}")
    
    plt.close()


def plot_confusion_matrix(cm, classes, output_path=None, normalize=False):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix (numpy array)
        classes: List of class names
        output_path: Path to save figure
        normalize: Normalize values
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved confusion matrix to: {output_path}")
    
    plt.close()


def plot_roc_curve(fpr, tpr, auc_score, output_path=None):
    """
    Plot ROC curve
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: Area under curve
        output_path: Path to save figure
    """
    plt.figure(figsize=(8, 6))
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved ROC curve to: {output_path}")
    
    plt.close()


def plot_image_grid(images, title=None, output_path=None, nrow=8):
    """
    Plot grid of images
    
    Args:
        images: Tensor or list of images
        title: Figure title
        output_path: Path to save figure
        nrow: Number of images per row
    """
    import torch
    from torchvision.utils import make_grid
    
    if isinstance(images, torch.Tensor):
        if images.dim() == 4:
            grid = make_grid(images, nrow=nrow, normalize=True, padding=2)
            grid = grid.permute(1, 2, 0).cpu().numpy()
        else:
            grid = images.cpu().numpy()
    else:
        grid = images
    
    plt.figure(figsize=(12, 12))
    
    if grid.shape[-1] == 1:
        plt.imshow(grid.squeeze(), cmap='gray')
    else:
        plt.imshow(grid)
    
    if title:
        plt.title(title, fontsize=14)
    
    plt.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved image grid to: {output_path}")
    
    plt.close()


def create_epoch_progression_plot(image_paths, epochs, output_path=None):
    """
    Create comparison plot of generated images across epochs
    
    Args:
        image_paths: List of image paths (one per epoch)
        epochs: List of epoch numbers
        output_path: Path to save figure
    """
    n_images = len(image_paths)
    fig, axes = plt.subplots(1, n_images, figsize=(4 * n_images, 4))
    
    if n_images == 1:
        axes = [axes]
    
    for ax, img_path, epoch in zip(axes, image_paths, epochs):
        img = plt.imread(img_path)
        ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        ax.set_title(f'Epoch {epoch}')
        ax.axis('off')
    
    plt.suptitle('Training Progression', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved epoch progression to: {output_path}")
    
    plt.close()
