# Grad-CAM Visualizations

This folder contains Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations that explain the model's predictions.

## What is Grad-CAM?

Grad-CAM uses the gradients flowing into the final convolutional layer to produce a coarse localization map highlighting important regions in the image for predicting the target class.

## Sample Visualizations

Each visualization contains:
1. **Original Image** - The input chest X-ray
2. **Heatmap** - Grad-CAM activation map showing important regions
3. **Overlay** - Heatmap superimposed on the original image

## Interpretation

- **Red/Yellow regions**: High importance for the prediction
- **Blue/Green regions**: Low importance
- **Lung fields**: Model correctly focuses on lung areas for pneumonia detection
- **Consolidation areas**: Model identifies opacity patterns indicative of pneumonia

## Key Observations

1. The model focuses on lower lung lobes where pneumonia typically manifests
2. Minimal attention to non-diagnostic areas (borders, artifacts)
3. Bilateral focus for severe cases
4. Appropriate attention drift based on pathology location
