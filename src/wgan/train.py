"""
WGAN-GP Training Script
Train Wasserstein GAN with Gradient Penalty on chest X-ray images
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.utils as vutils
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wgan.model import Generator, Critic, gradient_penalty


def get_dataloader(data_path, image_size=256, batch_size=32, subset_size=None, pneumonia_only=True):
    """
    Create dataloader for chest X-ray images
    
    Args:
        data_path: Path to data folder
        image_size: Target image size
        batch_size: Batch size
        subset_size: Optional subset size for faster training
        pneumonia_only: If True, only use PNEUMONIA class
    """
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    full_dataset = datasets.ImageFolder(root=data_path, transform=transform)
    print(f"Class mapping: {full_dataset.class_to_idx}")
    
    if pneumonia_only and 'PNEUMONIA' in full_dataset.class_to_idx:
        # Filter only PNEUMONIA samples
        pneumonia_idx = full_dataset.class_to_idx['PNEUMONIA']
        indices = [i for i, (_, label) in enumerate(full_dataset.samples) 
                   if label == pneumonia_idx]
        dataset = Subset(full_dataset, indices[:subset_size] if subset_size else indices)
        print(f"Using {len(dataset)} PNEUMONIA images")
    else:
        if subset_size:
            dataset = Subset(full_dataset, range(min(subset_size, len(full_dataset))))
        else:
            dataset = full_dataset
        print(f"Using {len(dataset)} images")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return dataloader


def train_wgan_gp(
    data_path,
    output_dir='outputs/wgan',
    latent_dim=100,
    feature_maps=128,
    image_size=256,
    batch_size=32,
    epochs=200,
    lr=1e-4,
    beta1=0.0,
    beta2=0.9,
    n_critic=5,
    lambda_gp=10,
    subset_size=2000,
    device='cuda'
):
    """
    Train WGAN-GP model
    
    Args:
        data_path: Path to training data
        output_dir: Directory to save outputs
        latent_dim: Size of latent vector
        feature_maps: Base feature maps for G and C
        image_size: Size of generated images
        batch_size: Training batch size
        epochs: Number of training epochs
        lr: Learning rate
        beta1, beta2: Adam optimizer betas
        n_critic: Critic updates per generator update
        lambda_gp: Gradient penalty coefficient
        subset_size: Number of images to use (None for all)
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
    dataloader = get_dataloader(data_path, image_size, batch_size, subset_size)
    
    # Initialize models
    G = Generator(latent_dim, feature_maps, channels=1).to(device)
    C = Critic(feature_maps, channels=1).to(device)
    
    # Optimizers
    optimizerG = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerC = torch.optim.Adam(C.parameters(), lr=lr, betas=(beta1, beta2))
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)
    
    # Training history
    G_losses = []
    C_losses = []
    
    print("\n🚀 Starting WGAN-GP Training...")
    print("=" * 60)
    print(f"Epochs: {epochs} | Batch Size: {batch_size}")
    print(f"n_critic: {n_critic} | Lambda GP: {lambda_gp}")
    print("=" * 60)
    
    for epoch in range(1, epochs + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        epoch_loss_G = 0.0
        epoch_loss_C = 0.0
        num_batches = 0
        
        for real_images, _ in pbar:
            real_images = real_images.to(device)
            batch_size_actual = real_images.size(0)
            
            # ==================
            # Train Critic
            # ==================
            for _ in range(n_critic):
                # Generate fake images
                z = torch.randn(batch_size_actual, latent_dim, 1, 1, device=device)
                fake_images = G(z).detach()
                
                # Critic scores
                C_real = C(real_images)
                C_fake = C(fake_images)
                
                # Gradient penalty
                gp = gradient_penalty(C, real_images, fake_images, device)
                
                # WGAN-GP loss
                loss_C = C_fake.mean() - C_real.mean() + lambda_gp * gp
                
                optimizerC.zero_grad()
                loss_C.backward()
                optimizerC.step()
            
            # ==================
            # Train Generator
            # ==================
            z = torch.randn(batch_size_actual, latent_dim, 1, 1, device=device)
            fake_images = G(z)
            
            # Generator loss (maximize critic score on fakes)
            loss_G = -C(fake_images).mean()
            
            optimizerG.zero_grad()
            loss_G.backward()
            optimizerG.step()
            
            # Track losses
            epoch_loss_C += loss_C.item()
            epoch_loss_G += loss_G.item()
            num_batches += 1
            
            pbar.set_postfix({
                'C_loss': f'{loss_C.item():.4f}',
                'G_loss': f'{loss_G.item():.4f}',
                'C_real': f'{C_real.mean().item():.4f}',
                'C_fake': f'{C_fake.mean().item():.4f}'
            })
        
        # Average losses
        avg_loss_C = epoch_loss_C / num_batches
        avg_loss_G = epoch_loss_G / num_batches
        C_losses.append(avg_loss_C)
        G_losses.append(avg_loss_G)
        
        print(f"Epoch {epoch}/{epochs} | C Loss: {avg_loss_C:.4f} | G Loss: {avg_loss_G:.4f}")
        
        # Save samples every 5 epochs
        if epoch % 5 == 0:
            G.eval()
            with torch.no_grad():
                samples = G(fixed_noise).detach().cpu()
            samples = (samples + 1) / 2.0  # Denormalize
            sample_path = os.path.join(output_dir, 'samples', f'epoch_{epoch:03d}.png')
            vutils.save_image(samples, sample_path, nrow=8, padding=2)
            print(f"💾 Saved samples to {sample_path}")
            G.train()
        
        # Save checkpoint every 20 epochs
        if epoch % 20 == 0:
            checkpoint_path = os.path.join(output_dir, 'checkpoints', f'checkpoint_epoch_{epoch:03d}.pth')
            torch.save({
                'epoch': epoch,
                'G_state_dict': G.state_dict(),
                'C_state_dict': C.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerC_state_dict': optimizerC.state_dict(),
                'G_losses': G_losses,
                'C_losses': C_losses,
            }, checkpoint_path)
            print(f"💾 Saved checkpoint to {checkpoint_path}")
    
    # Save final models
    torch.save(G.state_dict(), os.path.join(output_dir, 'generator_final.pth'))
    torch.save(C.state_dict(), os.path.join(output_dir, 'critic_final.pth'))
    
    print("\n✅ WGAN-GP Training Complete!")
    print(f"📁 Outputs saved to: {output_dir}")
    
    return G, C, G_losses, C_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train WGAN-GP on Chest X-Rays')
    parser.add_argument('--data_path', type=str, default='data/chest_xray/train',
                        help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='outputs/wgan',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='Latent vector dimension')
    parser.add_argument('--n_critic', type=int, default=5,
                        help='Number of critic updates per generator update')
    parser.add_argument('--lambda_gp', type=float, default=10,
                        help='Gradient penalty coefficient')
    parser.add_argument('--subset_size', type=int, default=2000,
                        help='Number of images to use (0 for all)')
    
    args = parser.parse_args()
    
    train_wgan_gp(
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        n_critic=args.n_critic,
        lambda_gp=args.lambda_gp,
        subset_size=args.subset_size if args.subset_size > 0 else None
    )
