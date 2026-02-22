"""
Evaluation Script for Pneumonia Classifier
Compute comprehensive metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier.model import PneumoniaClassifier


def evaluate_classifier(
    model_path,
    data_path,
    output_dir='outputs/evaluation',
    batch_size=16,
    device='cuda'
):
    """
    Comprehensive evaluation of pneumonia classifier
    
    Args:
        model_path: Path to trained model checkpoint
        data_path: Path to dataset (with test folder)
        output_dir: Directory to save evaluation results
        batch_size: Evaluation batch size
        device: 'cuda' or 'cpu'
    
    Returns:
        dict: Evaluation metrics
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint.get('classes', ['NORMAL', 'PNEUMONIA'])
    
    model = PneumoniaClassifier(num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from: {model_path}")
    print(f"Classes: {classes}")
    
    # Prepare data
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dir = os.path.join(data_path, 'test')
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Test set size: {len(test_dataset)}")
    
    # Collect predictions
    all_labels = []
    all_preds = []
    all_probs = []
    
    print("\n📊 Running evaluation...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    # ROC-AUC (for binary classification)
    if len(classes) == 2:
        roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs[:, 1])
    else:
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        fpr, tpr, thresholds = None, None, None
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print results
    print("\n" + "=" * 60)
    print("📋 EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\n🎯 Overall Accuracy: {accuracy:.2f}%")
    print(f"📈 ROC-AUC Score: {roc_auc:.4f}")
    print(f"\n📊 Macro Averages:")
    print(f"   Precision: {macro_precision:.4f}")
    print(f"   Recall: {macro_recall:.4f}")
    print(f"   F1-Score: {macro_f1:.4f}")
    
    print(f"\n📊 Per-Class Metrics:")
    print("-" * 50)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 50)
    
    for i, cls in enumerate(classes):
        print(f"{cls:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    
    print("-" * 50)
    print(f"{'Weighted Avg':<15} {weighted_precision:<12.4f} {weighted_recall:<12.4f} {weighted_f1:<12.4f} {sum(support):<10}")
    
    # Create confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f}%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    print(f"\n💾 Saved confusion matrix to {output_dir}/confusion_matrix.png")
    
    # Create ROC curve plot (for binary classification)
    if fpr is not None:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
        print(f"💾 Saved ROC curve to {output_dir}/roc_curve.png")
    
    # Save metrics to JSON
    metrics = {
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc),
        'macro': {
            'precision': float(macro_precision),
            'recall': float(macro_recall),
            'f1_score': float(macro_f1)
        },
        'weighted': {
            'precision': float(weighted_precision),
            'recall': float(weighted_recall),
            'f1_score': float(weighted_f1)
        },
        'per_class': {
            cls: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
            for i, cls in enumerate(classes)
        },
        'confusion_matrix': cm.tolist()
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Saved metrics to {output_dir}/metrics.json")
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=classes)
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
    print(f"💾 Saved classification report to {output_dir}/classification_report.txt")
    
    print("\n✅ Evaluation complete!")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Pneumonia Classifier')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, default='data/chest_xray',
                        help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation',
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    
    args = parser.parse_args()
    
    evaluate_classifier(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
