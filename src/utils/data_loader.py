"""
Data Loading Utilities
Common data loading functions for chest X-ray datasets
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image


def get_train_transforms(image_size=224, augment=True):
    """
    Get training transforms with optional augmentation
    
    Args:
        image_size: Target image size
        augment: Apply data augmentation
    
    Returns:
        torchvision.transforms.Compose
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_val_transforms(image_size=224):
    """
    Get validation/test transforms (no augmentation)
    
    Args:
        image_size: Target image size
    
    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_gan_transforms(image_size=64, grayscale=True):
    """
    Get transforms for GAN training
    
    Args:
        image_size: Target image size
        grayscale: Convert to grayscale
    
    Returns:
        torchvision.transforms.Compose
    """
    transform_list = []
    
    if grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=1))
    
    transform_list.extend([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * (1 if grayscale else 3), [0.5] * (1 if grayscale else 3))
    ])
    
    return transforms.Compose(transform_list)


class SimpleImageDataset(Dataset):
    """
    Simple dataset for loading images from a flat directory
    """
    
    def __init__(self, root_dir, label=0, transform=None, extensions=('.png', '.jpg', '.jpeg')):
        """
        Args:
            root_dir: Directory containing images
            label: Label for all images
            transform: Optional transform
            extensions: Valid file extensions
        """
        self.root_dir = root_dir
        self.label = label
        self.transform = transform
        
        self.files = [
            os.path.join(root_dir, f) 
            for f in os.listdir(root_dir) 
            if f.lower().endswith(extensions)
        ]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.label


def create_dataloader(
    data_path,
    batch_size=32,
    transform=None,
    shuffle=True,
    num_workers=4,
    subset_size=None
):
    """
    Create dataloader from ImageFolder
    
    Args:
        data_path: Path to data folder
        batch_size: Batch size
        transform: Optional transform
        shuffle: Shuffle data
        num_workers: Number of data loading workers
        subset_size: Optional subset size
    
    Returns:
        DataLoader, Dataset
    """
    if transform is None:
        transform = get_val_transforms()
    
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    
    if subset_size:
        indices = list(range(min(subset_size, len(dataset))))
        dataset = Subset(dataset, indices)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader, dataset


def get_class_distribution(dataset):
    """
    Get class distribution from dataset
    
    Args:
        dataset: ImageFolder dataset
    
    Returns:
        dict: Class counts
    """
    class_counts = {}
    
    if hasattr(dataset, 'samples'):
        for _, label in dataset.samples:
            class_name = dataset.classes[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    elif hasattr(dataset, 'dataset'):
        # For Subset
        for idx in dataset.indices:
            _, label = dataset.dataset.samples[idx]
            class_name = dataset.dataset.classes[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    return class_counts
