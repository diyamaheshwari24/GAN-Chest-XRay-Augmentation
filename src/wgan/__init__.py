"""
WGAN-GP Module
Wasserstein GAN with Gradient Penalty for chest X-ray synthesis
"""

from .model import Generator, Critic, gradient_penalty, get_wgan_models

__all__ = ['Generator', 'Critic', 'gradient_penalty', 'get_wgan_models']
