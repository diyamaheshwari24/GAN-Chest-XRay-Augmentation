"""
DCGAN Image Generation Script
Generate synthetic chest X-ray images using trained DCGAN
"""

import os
import sys
import argparse
import torch
import torchvision.utils as vutils

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dcgan.model import Generator


def generate_images(
    checkpoint_path,
    output_dir='outputs/generated',
    num_images=100,
    latent_dim=100,
    feature_maps=64,
    device='cuda'
):
    """
    Generate synthetic X-ray images using trained generator
    
    Args:
        checkpoint_path: Path to trained generator weights
        output_dir: Directory to save generated images
        num_images: Number of images to generate
        latent_dim: Latent vector dimension (must match training)
        feature_maps: Feature maps (must match training)
        device: 'cuda' or 'cpu'
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load generator
    G = Generator(latent_dim, feature_maps, channels=1).to(device)
    G.load_state_dict(torch.load(checkpoint_path, map_location=device))
    G.eval()
    
    print(f"\n🎨 Generating {num_images} synthetic X-ray images...")
    
    # Generate images
    batch_size = 64
    generated_count = 0
    
    with torch.no_grad():
        while generated_count < num_images:
            current_batch = min(batch_size, num_images - generated_count)
            
            # Generate latent vectors
            z = torch.randn(current_batch, latent_dim, 1, 1, device=device)
            
            # Generate images
            fake_images = G(z)
            
            # Save individual images
            for i in range(current_batch):
                img_path = os.path.join(output_dir, f'fake_{generated_count + i + 1}.png')
                vutils.save_image(fake_images[i], img_path, normalize=True)
            
            generated_count += current_batch
            print(f"Generated {generated_count}/{num_images} images", end='\r')
    
    print(f"\n✅ Generated {num_images} images!")
    print(f"📁 Saved to: {output_dir}")
    
    # Also save a grid visualization
    z = torch.randn(64, latent_dim, 1, 1, device=device)
    with torch.no_grad():
        samples = G(z)
    grid_path = os.path.join(output_dir, 'sample_grid.png')
    vutils.save_image(samples, grid_path, normalize=True, nrow=8)
    print(f"📊 Sample grid saved to: {grid_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate X-ray images with trained DCGAN')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to generator checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs/generated',
                        help='Output directory')
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of images to generate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='Latent vector dimension')
    
    args = parser.parse_args()
    
    generate_images(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_images=args.num_images,
        latent_dim=args.latent_dim
    )
