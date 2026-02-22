"""
Pneumonia Classifier Model
DenseNet121-based classifier for NORMAL vs PNEUMONIA classification
"""

import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


class PneumoniaClassifier(nn.Module):
    """
    DenseNet121-based Pneumonia Classifier
    
    Uses pretrained ImageNet weights with a custom classification head.
    Supports both frozen backbone (transfer learning) and full fine-tuning.
    
    Input:  (batch, 3, 224, 224)
    Output: (batch, 2) - logits for [NORMAL, PNEUMONIA]
    """
    
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=False):
        super(PneumoniaClassifier, self).__init__()
        
        # Load pretrained DenseNet121
        if pretrained:
            self.model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        else:
            self.model = densenet121(weights=None)
        
        # Replace classifier head
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.model.features.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        return self.model(x)
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for fine-tuning"""
        for param in self.model.features.parameters():
            param.requires_grad = True
    
    def get_features(self, x):
        """Extract features before classification head"""
        features = self.model.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out


def get_classifier(num_classes=2, pretrained=True, freeze_backbone=False, device='cuda'):
    """
    Factory function to create classifier
    
    Args:
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze DenseNet backbone
        device: 'cuda' or 'cpu'
    
    Returns:
        PneumoniaClassifier model
    """
    model = PneumoniaClassifier(num_classes, pretrained, freeze_backbone).to(device)
    return model


if __name__ == "__main__":
    # Test the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = get_classifier(device=device)
    
    # Test forward pass
    x = torch.randn(4, 3, 224, 224, device=device)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test feature extraction
    features = model.get_features(x)
    print(f"Features shape: {features.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n✅ Classifier model initialized successfully!")
