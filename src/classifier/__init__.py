"""
Classifier Module
DenseNet121-based pneumonia classifier with Grad-CAM support
"""

from .model import PneumoniaClassifier, get_classifier

__all__ = ['PneumoniaClassifier', 'get_classifier']
