"""
WGAN-GP Model Architecture
Wasserstein GAN with Gradient Penalty for 256x256 X-ray synthesis
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    WGAN-GP Generator Network
    
    Generates 256x256 grayscale chest X-ray images from latent vectors.
    Uses 7 transposed convolution layers with batch normalization.
    
    Architecture:
        Input:  (batch, 100, 1, 1)
        Output: (batch, 1, 256, 256)
    """
    
    def __init__(self, latent_dim=100, feature_maps=128, channels=1):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.feature_maps = feature_maps
        self.channels = channels
        
        self.main = nn.Sequential(
            # Input: (batch, latent_dim, 1, 1)
            # Output: (batch, feature_maps*8, 4, 4)
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            
            # Output: (batch, feature_maps*4, 8, 8)
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            
            # Output: (batch, feature_maps*2, 16, 16)
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            
            # Output: (batch, feature_maps, 32, 32)
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            
            # Output: (batch, feature_maps//2, 64, 64)
            nn.ConvTranspose2d(feature_maps, feature_maps // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps // 2),
            nn.ReLU(True),
            
            # Output: (batch, feature_maps//4, 128, 128)
            nn.ConvTranspose2d(feature_maps // 2, feature_maps // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps // 4),
            nn.ReLU(True),
            
            # Output: (batch, channels, 256, 256)
            nn.ConvTranspose2d(feature_maps // 4, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        return self.main(x)


class Critic(nn.Module):
    """
    WGAN-GP Critic (Discriminator) Network
    
    No sigmoid activation - outputs raw Wasserstein distance estimate.
    Uses batch normalization (can use layer norm for stricter WGAN).
    
    Architecture:
        Input:  (batch, 1, 256, 256)
        Output: (batch,) - Wasserstein critic score
    """
    
    def __init__(self, feature_maps=128, channels=1):
        super(Critic, self).__init__()
        
        self.feature_maps = feature_maps
        self.channels = channels
        
        self.main = nn.Sequential(
            # Input: (batch, channels, 256, 256)
            # Output: (batch, feature_maps//4, 128, 128)
            nn.Conv2d(channels, feature_maps // 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output: (batch, feature_maps//2, 64, 64)
            nn.Conv2d(feature_maps // 4, feature_maps // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps // 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output: (batch, feature_maps, 32, 32)
            nn.Conv2d(feature_maps // 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output: (batch, feature_maps*2, 16, 16)
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output: (batch, feature_maps*4, 8, 8)
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output: (batch, 1, 1, 1) - No sigmoid for WGAN
            nn.Conv2d(feature_maps * 4, 1, 4, 1, 0, bias=False),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        return self.main(x).view(-1)


def gradient_penalty(critic, real_images, fake_images, device):
    """
    Compute gradient penalty for WGAN-GP
    
    Args:
        critic: Critic network
        real_images: Batch of real images
        fake_images: Batch of generated images
        device: torch device
    
    Returns:
        Gradient penalty loss term
    """
    batch_size = real_images.size(0)
    
    # Random interpolation coefficient
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    epsilon = epsilon.expand_as(real_images)
    
    # Interpolated images
    interpolated = epsilon * real_images + (1 - epsilon) * fake_images
    interpolated.requires_grad_(True)
    
    # Critic output on interpolated images
    critic_interpolated = critic(interpolated)
    
    # Compute gradients
    grad_outputs = torch.ones_like(critic_interpolated, device=device)
    gradients = torch.autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Compute gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty


def get_wgan_models(latent_dim=100, feature_maps=128, channels=1, device='cuda'):
    """
    Factory function to create Generator and Critic
    
    Args:
        latent_dim: Size of latent vector
        feature_maps: Base number of feature maps
        channels: Number of image channels (1 for grayscale)
        device: 'cuda' or 'cpu'
    
    Returns:
        tuple: (generator, critic)
    """
    generator = Generator(latent_dim, feature_maps, channels).to(device)
    critic = Critic(feature_maps, channels).to(device)
    
    return generator, critic


if __name__ == "__main__":
    # Test the models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    G, C = get_wgan_models(device=device)
    
    # Test generator
    z = torch.randn(4, 100, 1, 1, device=device)
    fake_images = G(z)
    print(f"Generator output shape: {fake_images.shape}")
    
    # Test critic
    output = C(fake_images)
    print(f"Critic output shape: {output.shape}")
    
    # Test gradient penalty
    real = torch.randn(4, 1, 256, 256, device=device)
    gp = gradient_penalty(C, real, fake_images, device)
    print(f"Gradient penalty: {gp.item():.4f}")
    
    print("\n✅ WGAN-GP models initialized successfully!")
