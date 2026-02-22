# Grad-CAM Visualizations

This directory contains Grad-CAM visualization outputs from the pneumonia classifier.

## Contents

- Heatmap overlays on test images
- Top prediction visualizations
- Comparative analysis between NORMAL and PNEUMONIA cases

## Sample Visualizations

After running Grad-CAM analysis, this directory will contain:

```
gradcam/
├── normal/
│   ├── normal_001_gradcam.png
│   ├── normal_002_gradcam.png
│   └── ...
├── pneumonia/
│   ├── pneumonia_001_gradcam.png
│   ├── pneumonia_002_gradcam.png
│   └── ...
├── misclassified/
│   ├── false_positive_001.png
│   └── false_negative_001.png
└── summary/
    ├── top5_gradcam_grid.png
    └── gradcam_statistics.csv
```

## Generating Visualizations

```bash
python src/classifier/gradcam.py \
    --model_path models/pneumonia_densenet.pth \
    --data_path data/test \
    --output results/gradcam/
```

## Interpretation Guide

### Reading Grad-CAM Heatmaps

- **Red/Yellow regions**: High importance for prediction
- **Blue/Green regions**: Lower importance
- **Overlay transparency**: Original image with heatmap overlay

### Expected Patterns

**PNEUMONIA cases:**
- Attention focused on areas of consolidation
- Highlights opacity regions in lungs
- May show bilateral or unilateral patterns

**NORMAL cases:**
- More distributed attention
- Clear lung fields with minimal hot spots
- Focus on overall lung structure

## Quality Metrics

| Metric | Value |
|--------|-------|
| Images analyzed | 624 |
| Average confidence (correct) | 0.89 |
| Average confidence (incorrect) | 0.62 |
