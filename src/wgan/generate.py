"""
WGAN-GP Image Generation Script
Generate synthetic chest X-ray images using trained WGAN-GP
"""

import os
import sys
import argparse
import torch
import torchvision.utils as vutils
from PIL import Image, ImageFilter
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wgan.model import Generator


def upscale_image(image_tensor, target_size=512, sharpen=True):
    """
    Upscale image with high-quality interpolation
    
    Args:
        image_tensor: Single image tensor (C, H, W)
        target_size: Target resolution
        sharpen: Apply unsharp mask for edge enhancement
    
    Returns:
        Upscaled PIL Image
    """
    # Denormalize
    img = (image_tensor + 1) / 2.0
    img = img.clamp(0, 1)
    
    # Convert to PIL
    img = img.squeeze(0).cpu().numpy()
    img = (img * 255).astype('uint8')
    pil_img = Image.fromarray(img, mode='L').convert('RGB')
    
    # High-quality resize
    pil_img = pil_img.resize((target_size, target_size), resample=Image.LANCZOS)
    
    # Optional sharpening
    if sharpen:
        pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    
    return pil_img


def generate_images(
    checkpoint_path,
    output_dir='outputs/generated',
    num_images=2000,
    latent_dim=100,
    feature_maps=128,
    upscale=False,
    upscale_size=512,
    device='cuda'
):
    """
    Generate synthetic X-ray images using trained WGAN-GP generator
    
    Args:
        checkpoint_path: Path to trained generator weights
        output_dir: Directory to save generated images
        num_images: Number of images to generate
        latent_dim: Latent vector dimension (must match training)
        feature_maps: Feature maps (must match training)
        upscale: Whether to upscale images to higher resolution
        upscale_size: Target upscale resolution
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
        pbar = tqdm(total=num_images, desc="Generating")
        
        while generated_count < num_images:
            current_batch = min(batch_size, num_images - generated_count)
            
            # Generate latent vectors
            z = torch.randn(current_batch, latent_dim, 1, 1, device=device)
            
            # Generate images
            fake_images = G(z)
            
            # Save individual images
            for i in range(current_batch):
                img_idx = generated_count + i + 1
                
                if upscale:
                    # Upscale and save
                    pil_img = upscale_image(fake_images[i], upscale_size)
                    img_path = os.path.join(output_dir, f'fake_{img_idx}.png')
                    pil_img.save(img_path, quality=95)
                else:
                    # Save at original resolution
                    img_path = os.path.join(output_dir, f'fake_{img_idx}.png')
                    vutils.save_image((fake_images[i] + 1) / 2.0, img_path)
            
            generated_count += current_batch
            pbar.update(current_batch)
        
        pbar.close()
    
    print(f"\n✅ Generated {num_images} images!")
    print(f"📁 Saved to: {output_dir}")
    
    # Save a grid visualization
    z = torch.randn(64, latent_dim, 1, 1, device=device)
    with torch.no_grad():
        samples = G(z)
    samples = (samples + 1) / 2.0  # Denormalize
    grid_path = os.path.join(output_dir, 'sample_grid.png')
    vutils.save_image(samples, grid_path, nrow=8, padding=2)
    print(f"📊 Sample grid saved to: {grid_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate X-ray images with trained WGAN-GP')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to generator checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs/generated',
                        help='Output directory')
    parser.add_argument('--num_images', type=int, default=2000,
                        help='Number of images to generate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='Latent vector dimension')
    parser.add_argument('--upscale', action='store_true',
                        help='Upscale images to 512x512')
    parser.add_argument('--upscale_size', type=int, default=512,
                        help='Upscale target size')
    
    args = parser.parse_args()
    
    generate_images(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_images=args.num_images,
        latent_dim=args.latent_dim,
        upscale=args.upscale,
        upscale_size=args.upscale_size
    )
