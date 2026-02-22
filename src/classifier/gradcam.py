"""
Grad-CAM Implementation for Pneumonia Classifier
Generate visual explanations for model predictions
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier.model import PneumoniaClassifier


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping
    
    Visualizes which regions of an image are important for classification.
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: Trained classifier model
            target_layer: Layer to extract activations from
        """
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Class index to generate CAM for (None = predicted class)
        
        Returns:
            cam: Grad-CAM heatmap (H, W)
            pred_class: Predicted class index
            confidence: Prediction confidence
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        confidence = probs[0, target_class].item()
        
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Compute Grad-CAM
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, target_class, confidence


def generate_gradcam(
    model_path,
    image_path,
    output_path=None,
    device='cuda'
):
    """
    Generate Grad-CAM visualization for a single image
    
    Args:
        model_path: Path to trained model
        image_path: Path to input image
        output_path: Path to save visualization (optional)
        device: 'cuda' or 'cpu'
    
    Returns:
        dict: Results including prediction, confidence, and paths
    """
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint.get('classes', ['NORMAL', 'PNEUMONIA'])
    
    model = PneumoniaClassifier(num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Get target layer (last conv layer in DenseNet)
    target_layer = model.model.features[-1]
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    original_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Generate Grad-CAM
    cam, pred_class, confidence = grad_cam.generate(input_tensor)
    
    # Resize CAM to image size
    cam_resized = cv2.resize(cam, (224, 224))
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    # Convert original image
    original_resized = original_image.resize((224, 224))
    original_array = np.array(original_resized) / 255.0
    
    # Overlay
    overlay = heatmap * 0.4 + original_array * 0.6
    overlay = np.clip(overlay, 0, 1)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original image
    axes[0].imshow(original_resized)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title(f'Prediction: {classes[pred_class]}\nConfidence: {confidence:.2%}')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved Grad-CAM visualization to: {output_path}")
    
    plt.close()
    
    return {
        'predicted_class': classes[pred_class],
        'confidence': confidence,
        'cam': cam_resized,
        'overlay': overlay
    }


def batch_gradcam(
    model_path,
    image_dir,
    output_dir,
    num_images=5,
    device='cuda'
):
    """
    Generate Grad-CAM visualizations for multiple images
    
    Args:
        model_path: Path to trained model
        image_dir: Directory containing images
        output_dir: Directory to save visualizations
        num_images: Number of images to process
        device: 'cuda' or 'cpu'
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image files
    image_files = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, f))
    
    image_files = image_files[:num_images]
    
    print(f"\n🔍 Generating Grad-CAM for {len(image_files)} images...")
    
    results = []
    for i, img_path in enumerate(image_files):
        output_path = os.path.join(output_dir, f'gradcam_{i+1}.png')
        result = generate_gradcam(model_path, img_path, output_path, device)
        result['image_path'] = img_path
        results.append(result)
        print(f"  [{i+1}/{len(image_files)}] {os.path.basename(img_path)}: "
              f"{result['predicted_class']} ({result['confidence']:.2%})")
    
    print(f"\n✅ Saved {len(results)} Grad-CAM visualizations to: {output_dir}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualizations')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--image_path', type=str,
                        help='Path to single image')
    parser.add_argument('--image_dir', type=str,
                        help='Directory of images for batch processing')
    parser.add_argument('--output_dir', type=str, default='outputs/gradcam',
                        help='Output directory')
    parser.add_argument('--num_images', type=int, default=5,
                        help='Number of images for batch processing')
    
    args = parser.parse_args()
    
    if args.image_path:
        output_path = os.path.join(args.output_dir, 'gradcam_single.png')
        os.makedirs(args.output_dir, exist_ok=True)
        generate_gradcam(args.model_path, args.image_path, output_path)
    elif args.image_dir:
        batch_gradcam(args.model_path, args.image_dir, args.output_dir, args.num_images)
    else:
        print("Please provide either --image_path or --image_dir")
