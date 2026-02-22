# Model Architecture Details

This document provides in-depth technical details about the neural network architectures used in this project.

## Table of Contents

1. [DCGAN Architecture](#dcgan-architecture)
2. [WGAN-GP Architecture](#wgan-gp-architecture)
3. [DenseNet121 Classifier](#densenet121-classifier)
4. [Grad-CAM Implementation](#grad-cam-implementation)

---

## DCGAN Architecture

### Overview

Deep Convolutional Generative Adversarial Network (DCGAN) was introduced by Radford et al. in 2015. It applies convolutional neural networks to GANs with specific architectural constraints for stable training.

### Generator

```
Input: Latent Vector z ~ N(0, 1) of dimension (100, 1, 1)

Layer 1: ConvTranspose2d(100 → 512, k=4, s=1, p=0) + BatchNorm + ReLU → (512, 4, 4)
Layer 2: ConvTranspose2d(512 → 256, k=4, s=2, p=1) + BatchNorm + ReLU → (256, 8, 8)
Layer 3: ConvTranspose2d(256 → 128, k=4, s=2, p=1) + BatchNorm + ReLU → (128, 16, 16)
Layer 4: ConvTranspose2d(128 → 64, k=4, s=2, p=1) + BatchNorm + ReLU → (64, 32, 32)
Layer 5: ConvTranspose2d(64 → 1, k=4, s=2, p=1) + Tanh → (1, 64, 64)

Output: Grayscale image of size (1, 64, 64) in range [-1, 1]
```

### Discriminator

```
Input: Grayscale image of size (1, 64, 64)

Layer 1: Conv2d(1 → 64, k=4, s=2, p=1) + LeakyReLU(0.2) → (64, 32, 32)
Layer 2: Conv2d(64 → 128, k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2) → (128, 16, 16)
Layer 3: Conv2d(128 → 256, k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2) → (256, 8, 8)
Layer 4: Conv2d(256 → 512, k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2) → (512, 4, 4)
Layer 5: Conv2d(512 → 1, k=4, s=1, p=0) + Sigmoid → (1, 1, 1)

Output: Probability scalar in range [0, 1]
```

### Training Details

- **Loss Function**: Binary Cross-Entropy (BCE)
- **Optimizer**: Adam (β₁=0.5, β₂=0.999)
- **Learning Rate**: 0.0002
- **Batch Size**: 64
- **Epochs**: 50

---

## WGAN-GP Architecture

### Overview

Wasserstein GAN with Gradient Penalty (WGAN-GP) by Gulrajani et al. addresses training instability in traditional GANs by using the Wasserstein distance and gradient penalty instead of BCE loss.

### Generator (256×256)

```
Input: Latent Vector z ~ N(0, 1) of dimension (100, 1, 1)

Layer 1: ConvTranspose2d(100 → 1024, k=4, s=1, p=0) + BatchNorm + ReLU → (1024, 4, 4)
Layer 2: ConvTranspose2d(1024 → 512, k=4, s=2, p=1) + BatchNorm + ReLU → (512, 8, 8)
Layer 3: ConvTranspose2d(512 → 256, k=4, s=2, p=1) + BatchNorm + ReLU → (256, 16, 16)
Layer 4: ConvTranspose2d(256 → 128, k=4, s=2, p=1) + BatchNorm + ReLU → (128, 32, 32)
Layer 5: ConvTranspose2d(128 → 64, k=4, s=2, p=1) + BatchNorm + ReLU → (64, 64, 64)
Layer 6: ConvTranspose2d(64 → 32, k=4, s=2, p=1) + BatchNorm + ReLU → (32, 128, 128)
Layer 7: ConvTranspose2d(32 → 1, k=4, s=2, p=1) + Tanh → (1, 256, 256)

Output: Grayscale image of size (1, 256, 256) in range [-1, 1]
```

### Critic (Discriminator)

```
Input: Grayscale image of size (1, 256, 256)

Layer 1: Conv2d(1 → 32, k=4, s=2, p=1) + LeakyReLU(0.2) → (32, 128, 128)
Layer 2: Conv2d(32 → 64, k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2) → (64, 64, 64)
Layer 3: Conv2d(64 → 128, k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2) → (128, 32, 32)
Layer 4: Conv2d(128 → 256, k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2) → (256, 16, 16)
Layer 5: Conv2d(256 → 512, k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2) → (512, 8, 8)
Layer 6: Conv2d(512 → 1, k=4, s=1, p=0) → (1, 1, 1)

Output: Wasserstein critic score (unbounded)
```

### WGAN-GP Loss

```
L_critic = E[D(G(z))] - E[D(x)] + λ * GP

where GP = E[(||∇_x̂ D(x̂)||₂ - 1)²]
and x̂ = εx + (1-ε)G(z), ε ~ U(0,1)
```

### Training Details

- **Loss Function**: Wasserstein Loss + Gradient Penalty
- **Optimizer**: Adam (β₁=0.0, β₂=0.9)
- **Learning Rate**: 1e-4
- **n_critic**: 5 (critic updates per generator update)
- **λ_gp**: 10 (gradient penalty coefficient)
- **Batch Size**: 32
- **Epochs**: 200

---

## DenseNet121 Classifier

### Overview

DenseNet121 (Densely Connected Convolutional Network) connects each layer to every other layer in a feed-forward fashion. This architecture encourages feature reuse and strengthens feature propagation.

### Architecture

```
Input: RGB image of size (3, 224, 224)

Stem:
  Conv2d(3 → 64, k=7, s=2, p=3) + BatchNorm + ReLU + MaxPool → (64, 56, 56)

Dense Block 1: 6 layers → (256, 56, 56)
Transition 1: Conv + AvgPool → (128, 28, 28)

Dense Block 2: 12 layers → (512, 28, 28)
Transition 2: Conv + AvgPool → (256, 14, 14)

Dense Block 3: 24 layers → (1024, 14, 14)
Transition 3: Conv + AvgPool → (512, 7, 7)

Dense Block 4: 16 layers → (1024, 7, 7)

Classification Head (Modified):
  BatchNorm + ReLU + AdaptiveAvgPool → (1024, 1, 1)
  Flatten → (1024,)
  Dropout(0.3) + Linear(1024 → 512) + ReLU → (512,)
  Dropout(0.3) + Linear(512 → 2) → (2,)

Output: Class logits for [NORMAL, PNEUMONIA]
```

### Training Details

- **Pretrained Weights**: ImageNet
- **Optimizer**: AdamW (weight decay = 1e-4)
- **Learning Rate**: 1e-4 with ReduceLROnPlateau
- **Batch Size**: 16
- **Epochs**: 25 (early stopping with patience=5)
- **Class Balancing**: Weighted Random Sampler + Class-weighted Loss
- **Data Augmentation**: RandomCrop, HorizontalFlip, Rotation, ColorJitter

---

## Grad-CAM Implementation

### Overview

Gradient-weighted Class Activation Mapping (Grad-CAM) produces visual explanations for CNN predictions by using the gradients of the target class flowing into the final convolutional layer.

### Algorithm

```python
# 1. Forward pass
features = model.features(input)  # Get last conv layer output
output = model(input)             # Get class predictions

# 2. Backward pass
loss = output[0, target_class]
loss.backward()

# 3. Compute Grad-CAM
gradients = get_gradients()       # ∂y^c / ∂A^k
weights = global_average_pool(gradients)  # α_k^c = (1/Z) Σᵢⱼ ∂y^c/∂A^k_ij

# 4. Weighted combination
cam = ReLU(Σ_k α_k^c * A^k)

# 5. Normalize and resize
cam = normalize(cam)
cam = resize(cam, input_size)
```

### Implementation Notes

- **Target Layer**: `model.features[-1]` (DenseBlock4)
- **Hook Registration**: Forward hook for activations, backward hook for gradients
- **Visualization**: Heatmap overlay using cv2.applyColorMap with COLORMAP_JET

---

## Parameter Counts

| Model | Total Parameters | Trainable Parameters |
|-------|-----------------|---------------------|
| DCGAN Generator | 3.5M | 3.5M |
| DCGAN Discriminator | 2.8M | 2.8M |
| WGAN-GP Generator | 14.2M | 14.2M |
| WGAN-GP Critic | 11.8M | 11.8M |
| DenseNet121 Classifier | 7.0M | 7.0M (or 0.5M if frozen) |

---

## References

1. Radford, A., et al. "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks." arXiv:1511.06434 (2015)
2. Gulrajani, I., et al. "Improved Training of Wasserstein GANs." arXiv:1704.00028 (2017)
3. Huang, G., et al. "Densely Connected Convolutional Networks." arXiv:1608.06993 (2017)
4. Selvaraju, R. R., et al. "Grad-CAM: Visual Explanations from Deep Networks." arXiv:1610.02391 (2017)
