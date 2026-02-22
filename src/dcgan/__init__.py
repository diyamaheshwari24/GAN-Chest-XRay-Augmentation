"""
DCGAN Module
Deep Convolutional Generative Adversarial Network for chest X-ray synthesis
"""

from .model import Generator, Discriminator, get_dcgan_models

__all__ = ['Generator', 'Discriminator', 'get_dcgan_models']
