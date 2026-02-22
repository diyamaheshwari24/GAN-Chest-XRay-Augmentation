"""
DCGAN Model Architecture
Deep Convolutional Generative Adversarial Network for 64x64 X-ray synthesis
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    DCGAN Generator Network
    
    Transforms a latent vector (100,1,1) into a 64x64 grayscale image
    using transposed convolutions with batch normalization.
    
    Architecture:
        Input:  (batch, 100, 1, 1)
        Output: (batch, 1, 64, 64)
    """
    
    def __init__(self, latent_dim=100, feature_maps=64, channels=1):
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
            
            # Output: (batch, channels, 64, 64)
            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),
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


class Discriminator(nn.Module):
    """
    DCGAN Discriminator Network
    
    Binary classifier that distinguishes real images from generated ones.
    Uses strided convolutions with leaky ReLU activations.
    
    Architecture:
        Input:  (batch, 1, 64, 64)
        Output: (batch,) - probability of being real
    """
    
    def __init__(self, feature_maps=64, channels=1):
        super(Discriminator, self).__init__()
        
        self.feature_maps = feature_maps
        self.channels = channels
        
        self.main = nn.Sequential(
            # Input: (batch, channels, 64, 64)
            # Output: (batch, feature_maps, 32, 32)
            nn.Conv2d(channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output: (batch, feature_maps*2, 16, 16)
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output: (batch, feature_maps*4, 8, 8)
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output: (batch, feature_maps*8, 4, 4)
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output: (batch, 1, 1, 1)
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
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


def get_dcgan_models(latent_dim=100, feature_maps=64, channels=1, device='cuda'):
    """
    Factory function to create Generator and Discriminator
    
    Args:
        latent_dim: Size of latent vector
        feature_maps: Base number of feature maps
        channels: Number of image channels (1 for grayscale)
        device: 'cuda' or 'cpu'
    
    Returns:
        tuple: (generator, discriminator)
    """
    generator = Generator(latent_dim, feature_maps, channels).to(device)
    discriminator = Discriminator(feature_maps, channels).to(device)
    
    return generator, discriminator


if __name__ == "__main__":
    # Test the models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    G, D = get_dcgan_models(device=device)
    
    # Test generator
    z = torch.randn(4, 100, 1, 1, device=device)
    fake_images = G(z)
    print(f"Generator output shape: {fake_images.shape}")
    
    # Test discriminator
    output = D(fake_images)
    print(f"Discriminator output shape: {output.shape}")
    
    print("\n✅ DCGAN models initialized successfully!")
