"""
Pneumonia Classifier Training Script
Train DenseNet121 classifier with proper data augmentation and class balancing
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier.model import PneumoniaClassifier


def get_transforms():
    """Get training and validation transforms with augmentation"""
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def get_weighted_sampler(dataset):
    """Create weighted sampler for class balancing"""
    
    # Count samples per class
    class_counts = {}
    for _, label in dataset.samples:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    # Compute weights
    total = sum(class_counts.values())
    class_weights = {cls: total / count for cls, count in class_counts.items()}
    
    # Assign weight to each sample
    sample_weights = [class_weights[label] for _, label in dataset.samples]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


def train_classifier(
    data_path,
    output_dir='outputs/classifier',
    epochs=25,
    batch_size=16,
    lr=1e-4,
    freeze_backbone=False,
    use_class_weights=True,
    patience=5,
    device='cuda'
):
    """
    Train pneumonia classifier with improved techniques
    
    Args:
        data_path: Path to dataset (with train/val/test folders)
        output_dir: Directory to save outputs
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        freeze_backbone: Freeze DenseNet backbone
        use_class_weights: Use class weights for imbalanced data
        patience: Early stopping patience
        device: 'cuda' or 'cpu'
    """
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Load datasets
    train_dir = os.path.join(data_path, 'train')
    test_dir = os.path.join(data_path, 'test')
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
    
    print(f"\nClasses: {train_dataset.classes}")
    print(f"Class mapping: {train_dataset.class_to_idx}")
    
    # Count class distribution
    train_class_counts = {}
    for _, label in train_dataset.samples:
        class_name = train_dataset.classes[label]
        train_class_counts[class_name] = train_class_counts.get(class_name, 0) + 1
    print(f"Training distribution: {train_class_counts}")
    
    # Create weighted sampler for balanced training
    sampler = get_weighted_sampler(train_dataset)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\nTrain size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    # Initialize model
    model = PneumoniaClassifier(
        num_classes=2, 
        pretrained=True, 
        freeze_backbone=freeze_backbone
    ).to(device)
    
    # Class weights for loss function
    if use_class_weights:
        # Inverse frequency weighting
        total = sum(train_class_counts.values())
        weights = []
        for cls in train_dataset.classes:
            weights.append(total / train_class_counts[cls])
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        class_weights = class_weights / class_weights.sum()  # Normalize
        print(f"Class weights: {dict(zip(train_dataset.classes, class_weights.cpu().tolist()))}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    print("\n🚀 Starting Training...")
    print("=" * 60)
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            train_correct += preds.eq(labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * train_correct / train_total:.2f}%'
            })
        
        avg_train_loss = train_loss / train_total
        avg_train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)
        
        avg_val_loss = val_loss / val_total
        avg_val_acc = 100. * val_correct / val_total
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        history['lr'].append(current_lr)
        
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': avg_val_acc,
                'val_loss': avg_val_loss,
                'classes': train_dataset.classes,
            }, os.path.join(output_dir, 'best_model.pth'))
            print(f"  ✅ New best model saved! (Val Acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️ Early stopping triggered at epoch {epoch}")
            break
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'classes': train_dataset.classes,
    }, os.path.join(output_dir, 'final_model.pth'))
    
    # Save training history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"📁 Outputs saved to: {output_dir}")
    
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Pneumonia Classifier')
    parser.add_argument('--data_path', type=str, default='data/chest_xray',
                        help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='outputs/classifier',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    
    args = parser.parse_args()
    
    train_classifier(
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience
    )
