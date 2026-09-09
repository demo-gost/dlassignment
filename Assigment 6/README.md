# 🍅 Assignment 6 — Building a CNN for Tomato Image Classification

## 📌 Objective

This lab marked my first step into **Computer Vision**! Instead of tabular data (like the Iris or Wine datasets), I trained a **Convolutional Neural Network (CNN)** to classify images of tomatoes into 4 categories: **Ripe, Unripe, Damaged, and Old**. I used the Kaggle `tomatoes-dataset` and built the CNN from scratch.

---

## 📂 Files

| File | Description |
|------|-------------|
| `lab6.ipynb` | Main notebook with dataset loading, CNN architecture, training, and evaluation |

---

## 📦 Dataset

| Detail | Value |
|--------|-------|
| Source | Kaggle — `enalis/tomatoes-dataset` (downloaded via `kagglehub`) |
| Classes | Damaged, Old, Ripe, Unripe |
| Training Images | **6,500** total |
| Validation Images | **724** total |
| Image Size (resized) | 96 × 96 pixels |

**Training Class Breakdown:**

| Class | Count |
|-------|-------|
| Damaged | 949 |
| Old | 1,992 |
| Ripe | 1,975 |
| Unripe | 1,584 |

**Validation Class Breakdown:**

| Class | Count |
|-------|-------|
| Damaged | 106 |
| Old | 222 |
| Ripe | 220 |
| Unripe | 176 |

---

## 📖 What I Did — Step by Step

### 1. 📥 Downloaded the Dataset from Kaggle
- Installed and used `kagglehub` to download the `enalis/tomatoes-dataset` directly into Colab
- Explored the folder structure: the dataset had `train/` and `val/` subfolders, each with 4 class sub-directories

### 2. 🔍 Explored & Visualized the Dataset
- Counted the total images per class in both train and validation sets
- Displayed sample images for each class (Damaged, Old, Ripe, Unripe) using Matplotlib

### 3. 🔄 Data Augmentation & Preprocessing with `ImageDataGenerator`
- **Training data** — applied augmentation to prevent overfitting:
  - Pixel rescaling (÷255)
  - Random rotation (±20°)
  - Width & height shifts (±10%)
  - Zoom (±15%)
  - Horizontal flipping
  - Fill mode: nearest
- **Validation data** — only rescaled (no augmentation, to get accurate evaluation)
- **Output:** Batches of shape `(32, 96, 96, 3)` with one-hot labels of shape `(32, 4)`

### 4. 🏗️ Designed the CNN Architecture from Scratch

```
Input: (96, 96, 3)
    ↓
Conv2D(16, 3×3, ReLU, same padding)  → MaxPooling2D(2×2)
    ↓
Conv2D(32, 3×3, ReLU, same padding)  → MaxPooling2D(2×2)
    ↓
Conv2D(64, 3×3, ReLU, same padding)  → MaxPooling2D(2×2)
    ↓
GlobalAveragePooling2D()
    ↓
Dense(64, ReLU)
    ↓
Dropout(0.3)
    ↓
Dense(4, Softmax)  ← 4 tomato classes
```

**Why this architecture?**
- 3 convolutional blocks progressively learn more complex features (edges → textures → shapes)
- `GlobalAveragePooling2D` is used instead of `Flatten` to reduce parameters and overfitting
- `Dropout(0.3)` adds extra regularization

### 5. ⚙️ Compiled & Trained the Model
- **Optimizer:** Adam (lr=0.001)
- **Loss:** Categorical Crossentropy (because labels are one-hot encoded)
- **Metric:** Accuracy
- **Epochs:** 10
- **Batch Size:** 32

---

## 🛠️ Libraries Used

```python
import numpy as np, matplotlib.pyplot as plt, seaborn as sns
import tensorflow as tf, os, kagglehub
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
```

---

## 💡 Key Takeaways

- Working with image data is **completely different** from tabular data — you need to handle file paths, loading images in batches, and pixel normalization.
- **Data augmentation** is critical when you don't have millions of images. By randomly flipping, rotating, and zooming training images, we artificially expand the dataset and make the model more robust.
- **CNNs naturally excel at images** because convolutional layers share weights and can detect the same pattern (like a tomato stem) anywhere in the image.
- `GlobalAveragePooling2D` is a great alternative to `Flatten` — it dramatically reduces the number of parameters while preserving spatial information.
- The class imbalance (Old/Ripe dominate the training set) could affect performance on the "Damaged" class — something to address with class weights in future experiments.
