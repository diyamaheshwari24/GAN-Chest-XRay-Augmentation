# Dataset

This directory should contain the Chest X-Ray Pneumonia dataset.

## Dataset Information

- **Source**: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Size**: ~1.2 GB
- **Images**: 5,856 total (5,232 training + 624 testing)
- **Classes**: NORMAL, PNEUMONIA

## Download Instructions

### Option 1: Kaggle CLI (Recommended)

```bash
# Install Kaggle CLI
pip install kaggle

# Configure API key (download from kaggle.com/account)
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# Extract to data directory
unzip chest-xray-pneumonia.zip -d data/
mv data/chest_xray/* data/
rm -rf data/chest_xray
rm chest-xray-pneumonia.zip
```

### Option 2: Manual Download

1. Go to [Kaggle Dataset Page](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Click "Download" (requires Kaggle account)
3. Extract the ZIP file
4. Place contents in this `data/` directory

## Expected Directory Structure

After downloading, the structure should be:

```
data/
├── README.md (this file)
├── train/
│   ├── NORMAL/
│   │   ├── IM-0115-0001.jpeg
│   │   ├── IM-0117-0001.jpeg
│   │   └── ... (1,341 images)
│   └── PNEUMONIA/
│       ├── person1_bacteria_1.jpeg
│       ├── person1_bacteria_2.jpeg
│       └── ... (3,875 images)
├── test/
│   ├── NORMAL/
│   │   └── ... (234 images)
│   └── PNEUMONIA/
│       └── ... (390 images)
└── val/
    ├── NORMAL/
    │   └── ... (8 images)
    └── PNEUMONIA/
        └── ... (8 images)
```

## Dataset Statistics

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1,341 | 3,875 | 5,216 |
| Test | 234 | 390 | 624 |
| Val | 8 | 8 | 16 |
| **Total** | **1,583** | **4,273** | **5,856** |

## Class Imbalance

The dataset has significant class imbalance:
- NORMAL: 27.0%
- PNEUMONIA: 73.0%

Our training scripts handle this using:
- Weighted Random Sampler
- Class-weighted loss function
- Data augmentation for minority class

## Image Properties

- **Format**: JPEG
- **Size**: Variable (typically 1000-2000 pixels)
- **Color**: Grayscale (stored as RGB in some cases)
- **Quality**: Chest X-ray radiographs

## Data Preprocessing

Our scripts automatically handle:
- Resizing to target dimensions (64×64 for DCGAN, 256×256 for WGAN, 224×224 for classifier)
- Normalization to [-1, 1] for GANs or ImageNet stats for classifier
- Grayscale conversion if needed

## Citation

```bibtex
@article{kermany2018identifying,
  title={Identifying medical diagnoses and treatable diseases by image-based deep learning},
  author={Kermany, Daniel S and Goldbaum, Michael and Cai, Wenjia and others},
  journal={Cell},
  volume={172},
  number={5},
  pages={1122--1131},
  year={2018},
  publisher={Elsevier}
}
```

## License

Please refer to the [Kaggle dataset page](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) for licensing information.
