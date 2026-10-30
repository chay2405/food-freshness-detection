# Model Directory

This directory contains the trained weights, architecture, and evaluation metrics for the Food Freshness Detection classifier.

## Files
- `food_freshness_model.keras` (~21.7 MB): Saved Keras model containing MobileNetV2 base weights, custom classification head, and optimizer state.
- `metrics.json`: High-level evaluation metrics on the held-out test set (480 images), used by the Streamlit application sidebar for real-time performance display.

## Model Details
- **Architecture**: MobileNetV2 transfer learning with ImageNet initialization
- **Input Dimensions**: 224 × 224 × 3 (RGB)
- **Output**: Single sigmoid unit representing $P(\text{Rotten}) \in [0.0, 1.0]$
  - Prediction < 0.5: **Fresh**
  - Prediction $\ge$ 0.5: **Rotten**
- **Test Accuracy**: 99.38% (ROC-AUC: 0.9997)

## Compatibility
If running in a local environment or server and encountering model deserialization warnings, ensure `tensorflow` or `tensorflow-cpu` version $\ge 2.16$ is installed (matching the version in `requirements.txt`).
