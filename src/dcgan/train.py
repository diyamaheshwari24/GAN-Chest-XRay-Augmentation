"""
DCGAN Training Script
Train DCGAN on chest X-ray images
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dcgan.model import Generator, Discriminator


def get_dataloader(data_path, image_size=64, batch_size=64):
    """Create dataloader for chest X-ray images"""
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return dataloader


def train_dcgan(
    data_path,
    output_dir='outputs/dcgan',
    latent_dim=100,
    feature_maps=64,
    image_size=64,
    batch_size=64,
    epochs=50,
    lr=0.0002,
    beta1=0.5,
    device='cuda'
):
    """
    Train DCGAN model
    
    Args:
        data_path: Path to training data
        output_dir: Directory to save outputs
        latent_dim: Size of latent vector
        feature_maps: Base feature maps for G and D
        image_size: Size of generated images
        batch_size: Training batch size
        epochs: Number of training epochs
        lr: Learning rate
        beta1: Adam optimizer beta1
        device: 'cuda' or 'cpu'
    """
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    dataloader = get_dataloader(data_path, image_size, batch_size)
    print(f"Loaded {len(dataloader.dataset)} images")
    
    # Initialize models
    G = Generator(latent_dim, feature_maps, channels=1).to(device)
    D = Discriminator(feature_maps, channels=1).to(device)
    
    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizerG = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)
    
    # Training loop
    G_losses = []
    D_losses = []
    
    print("\n🚀 Starting DCGAN Training...")
    print("=" * 50)
    
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        
        for i, (real_images, _) in enumerate(pbar):
            batch_size_actual = real_images.size(0)
            real_images = real_images.to(device)
            
            # Labels
            real_labels = torch.ones(batch_size_actual, device=device)
            fake_labels = torch.zeros(batch_size_actual, device=device)
            
            # ==================
            # Train Discriminator
            # ==================
            optimizerD.zero_grad()
            
            # Real images
            output_real = D(real_images)
            loss_D_real = criterion(output_real, real_labels)
            
            # Fake images
            noise = torch.randn(batch_size_actual, latent_dim, 1, 1, device=device)
            fake_images = G(noise)
            output_fake = D(fake_images.detach())
            loss_D_fake = criterion(output_fake, fake_labels)
            
            # Combined D loss
            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            optimizerD.step()
            
            # ==================
            # Train Generator
            # ==================
            optimizerG.zero_grad()
            
            output = D(fake_images)
            loss_G = criterion(output, real_labels)
            loss_G.backward()
            optimizerG.step()
            
            # Track losses
            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()
            
            pbar.set_postfix({
                'D_loss': f'{loss_D.item():.4f}',
                'G_loss': f'{loss_G.item():.4f}'
            })
        
        # Average losses
        avg_loss_D = epoch_loss_D / len(dataloader)
        avg_loss_G = epoch_loss_G / len(dataloader)
        G_losses.append(avg_loss_G)
        D_losses.append(avg_loss_D)
        
        print(f"Epoch {epoch+1}/{epochs} | D Loss: {avg_loss_D:.4f} | G Loss: {avg_loss_G:.4f}")
        
        # Save samples every 5 epochs
        if (epoch + 1) % 5 == 0:
            G.eval()
            with torch.no_grad():
                samples = G(fixed_noise).detach().cpu()
            sample_path = os.path.join(output_dir, 'samples', f'epoch_{epoch+1:03d}.png')
            vutils.save_image(samples, sample_path, normalize=True, nrow=8)
            print(f"💾 Saved samples to {sample_path}")
            G.train()
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                'G_losses': G_losses,
                'D_losses': D_losses,
            }, os.path.join(output_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1:03d}.pth'))
    
    # Save final models
    torch.save(G.state_dict(), os.path.join(output_dir, 'generator_final.pth'))
    torch.save(D.state_dict(), os.path.join(output_dir, 'discriminator_final.pth'))
    
    print("\n✅ Training Complete!")
    print(f"📁 Outputs saved to: {output_dir}")
    
    return G, D, G_losses, D_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DCGAN on Chest X-Rays')
    parser.add_argument('--data_path', type=str, default='data/chest_xray/train',
                        help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='outputs/dcgan',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='Latent vector dimension')
    
    args = parser.parse_args()
    
    train_dcgan(
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim
    )
