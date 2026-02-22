# Pre-trained Model Weights

This directory stores trained model weights. Due to file size limitations on GitHub, large model weights are not included in the repository.

## Available Models

| Model | File | Size | Download |
|-------|------|------|----------|
| DCGAN Generator | `dcgan_generator.pth` | ~14 MB | [Download](#) |
| DCGAN Discriminator | `dcgan_discriminator.pth` | ~11 MB | [Download](#) |
| WGAN-GP Generator | `wgan_generator.pth` | ~57 MB | [Download](#) |
| WGAN-GP Critic | `wgan_critic.pth` | ~47 MB | [Download](#) |
| DenseNet121 Classifier | `pneumonia_densenet.pth` | ~28 MB | [Download](#) |

## Download Instructions

### Option 1: Google Drive

Download pre-trained weights from Google Drive:

```
[Google Drive Link - To be added after upload]
```

After downloading, place the files in this `models/` directory.

### Option 2: Train from Scratch

If you prefer to train the models yourself:

```bash
# Train DCGAN
python src/dcgan/train.py --data_path data/train --epochs 50

# Train WGAN-GP
python src/wgan/train.py --data_path data/train --epochs 200

# Train Classifier
python src/classifier/train.py --data_path data --epochs 25
```

### Option 3: Hugging Face Hub (Recommended)

```python
# Coming soon - models will be uploaded to Hugging Face
from huggingface_hub import hf_hub_download

# Download classifier
model_path = hf_hub_download(
    repo_id="your-username/chest-xray-pneumonia",
    filename="pneumonia_densenet.pth"
)
```

## Loading Models

```python
import torch
from src.dcgan.model import DCGANGenerator
from src.wgan.model import WGANGenerator
from src.classifier.model import PneumoniaClassifier

# Load DCGAN Generator
dcgan_gen = DCGANGenerator(latent_dim=100, img_channels=1, feature_maps=64)
dcgan_gen.load_state_dict(torch.load('models/dcgan_generator.pth'))
dcgan_gen.eval()

# Load WGAN Generator
wgan_gen = WGANGenerator(latent_dim=100, img_channels=1, feature_maps=64)
wgan_gen.load_state_dict(torch.load('models/wgan_generator.pth'))
wgan_gen.eval()

# Load Classifier
classifier = PneumoniaClassifier(num_classes=2)
classifier.load_state_dict(torch.load('models/pneumonia_densenet.pth'))
classifier.eval()
```

## Model Checkpoints

During training, checkpoints are saved to this directory:

```
models/
├── dcgan_generator.pth      # Final DCGAN generator
├── dcgan_discriminator.pth  # Final DCGAN discriminator
├── wgan_generator.pth       # Final WGAN generator
├── wgan_critic.pth          # Final WGAN critic
├── pneumonia_densenet.pth   # Final classifier
└── checkpoints/             # Intermediate checkpoints
    ├── dcgan_epoch_10.pth
    ├── dcgan_epoch_20.pth
    ├── wgan_epoch_50.pth
    ├── wgan_epoch_100.pth
    └── ...
```

## File Integrity

After downloading, verify file integrity using MD5 checksums:

```bash
md5sum models/*.pth
```

Expected checksums (to be added after training):
```
[To be generated] dcgan_generator.pth
[To be generated] dcgan_discriminator.pth
[To be generated] wgan_generator.pth
[To be generated] wgan_critic.pth
[To be generated] pneumonia_densenet.pth
```

## Note on Reproducibility

Results may vary slightly due to:
- Random weight initialization
- GPU floating-point non-determinism
- Different PyTorch versions

For exact reproducibility, use the provided weights or set random seeds:

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```
