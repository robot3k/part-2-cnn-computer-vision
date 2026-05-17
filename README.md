# Part 2: Computer Vision Problem Formulation and CNN Prototype

## Overview

This project builds a Convolutional Neural Network (CNN) to classify manufacturing product images into four defect categories. The dataset simulates real industrial quality inspection — a camera photographs each product, and the AI instantly flags the defect type.

## Dataset

**Source:** Part 2 dataset (Masai Module 5 Assignment)  
**Link:** [Google Drive Dataset Folder](https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJIV-wBvUYs?usp=sharing)

| Class | Images | Description |
|-------|--------|-------------|
| `normal` | 120 | Clean product surface |
| `dent` | 120 | Circular dent-like marks |
| `scratch` | 120 | Linear scratch marks |
| `stain` | 120 | Colored stain patches |
| **Total** | **480** | Balanced — 25% per class |

- **Image size:** 64×64 pixels (after preprocessing) | **Channels:** RGB
- **Class balance:** Perfectly balanced — no augmentation needed for balance correction

## Problem Type: Image Classification

Each image belongs to exactly one class (one defect type). This is a **multi-class image classification** problem, not object detection (no bounding boxes needed) or segmentation (no pixel masks).

## CNN Architecture

```
Input: (64, 64, 3)
  → Conv2D(32 filters, 3×3, ReLU, padding='same')   # Detects edges/textures
  → MaxPooling2D(2×2)                                 # Spatial downsampling
  → Conv2D(64 filters, 3×3, ReLU, padding='same')   # Detects shapes
  → MaxPooling2D(2×2)
  → Conv2D(64 filters, 3×3, ReLU, padding='same')   # Detects defect patterns
  → Flatten
  → Dense(128, ReLU)
  → Dropout(0.4)
  → Dense(4, Softmax)                                 # 4-class output
```

- **Loss:** Sparse Categorical Cross-Entropy
- **Optimizer:** Adam
- **Epochs:** 30 | **Batch size:** 16 | **Validation split:** 15%

## Results

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| dent | 0.89 | 1.00 | 0.94 |
| normal | 1.00 | 1.00 | 1.00 |
| scratch | 1.00 | 0.96 | 0.98 |
| stain | 1.00 | 0.92 | 0.96 |
| **Overall** | **0.97** | **0.97** | **0.97** |

**Test Accuracy: 96.88%** — The CNN misclassified only 3 out of 96 test images.

## CNN Concepts Explained

**Convolution:** A filter slides across the image computing dot products — each filter learns to detect a specific visual feature (edge, corner, texture). Multiple filters run in parallel per layer.

**Pooling:** MaxPooling reduces spatial dimensions by 50%, making the model more efficient and translation-invariant. A dent detected 5 pixels to the left still produces the same pooled activation.

**ReLU:** Sets negatives to zero, introducing non-linearity that allows the network to learn curved, complex decision boundaries. Avoids vanishing gradients unlike sigmoid/tanh.

**CNNs vs. Feed-Forward Networks:** CNNs use parameter sharing (one filter applied across the whole image) and local connectivity — drastically fewer parameters than a fully-connected network on a 64×64×3 image (which would need 12,288 input neurons).

## Business Use Case: Manufacturing Quality Control

An AI-powered inspection system using this CNN can:
- Process **>10,000 units/hour** (vs. ~500 manually)
- Maintain **>96% detection accuracy** without fatigue
- Reduce warranty claims and customer returns
- Provide audit logs of every inspected unit

## Repository Structure

```
part-2-cnn-computer-vision/
├── README.md
├── notebook.ipynb
├── requirements.txt
├── sample_predictions/
│   └── prediction_outputs.png
└── results/
    ├── accuracy_loss_curves.png
    └── confusion_matrix.png
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

Place the `images/` folder (with subfolders `dent/`, `normal/`, `scratch/`, `stain/`) in the same directory as the notebook.
