# 🌱 Assignment 7 — Transfer Learning with VGG16, ResNet50 & EfficientNetB0

## 📌 Objective

Why build a model from scratch when you can stand on the shoulders of giants? In this lab, I applied **Transfer Learning** — a technique where you take a powerful model pre-trained on millions of images (ImageNet) and fine-tune it for a new, specific task. I used three industry-standard architectures — **VGG16, ResNet50, and EfficientNetB0** — to classify images from a Soyabean dataset.

---

## 📂 Files

| File | Description |
|------|-------------|
| `lab7.ipynb` | Main notebook with dataset loading, transfer learning experiments, and comparisons |

---

## 📦 Dataset

| Detail | Value |
|--------|-------|
| Source | Kaggle — `akansha03mulchandani/soyabean` (downloaded via `kagglehub`) |
| Dataset Format | VOC2007 format with JPEGImages & SegmentationClass |
| Total Images | 800 JPEG images |
| Classes Found | 3 (ImageSets, JPEGImages, SegmentationClass) |
| Train / Val Split | 80% / 20% (1,280 training / 320 validation after loading) |
| Image Size (resized) | 224 × 224 pixels (standard for ImageNet pre-trained models) |
| **TensorFlow Version** | 2.21.0 |

---

## 📖 What I Did — Step by Step

### 1. 📥 Downloaded the Soyabean Dataset from Kaggle
- Used `kagglehub` to download the `akansha03mulchandani/soyabean` dataset
- The dataset used **VOC2007 format** with folders: `JPEGImages/`, `SegmentationClass/`, `ImageSets/`
- Wrote helper code to automatically detect the correct image directory by checking which subfolders contain images

### 2. 🔍 Explored the Dataset Structure
- Counted images per class folder
- Identified that `JPEGImages` contained **800 images** and `SegmentationClass` contained **800 mask images**
- The model trained to classify using the folder structure

### 3. 📂 Loaded Data Using `image_dataset_from_directory`
- Used the modern `tf.keras.utils.image_dataset_from_directory()` (instead of the older `ImageDataGenerator`)
- Applied **80/20 validation split** automatically
- All images resized to **224 × 224** (required by VGG16 / ResNet50 / EfficientNetB0)
- Used `.prefetch(AUTOTUNE)` for efficient pipelining so the GPU doesn't have to wait for data

### 4. 🔄 Added On-the-Fly Data Augmentation
Applied augmentation as a Keras Sequential layer (applied only during training):
```python
tf.keras.layers.RandomFlip("horizontal")
tf.keras.layers.RandomRotation(0.1)
tf.keras.layers.RandomZoom(0.1)
```

### 5. 🏗️ Built a Reusable Transfer Learning Pipeline
Created a `create_transfer_model(base_model, model_name)` function that:
1. **Freezes** the pre-trained base model (no weight updates during initial training)
2. Applies appropriate **preprocessing** per model:
   - VGG16 → `vgg_preprocess`
   - ResNet50 → `resnet_preprocess`
   - EfficientNetB0 → has built-in preprocessing
3. Passes images through the frozen base model to extract features
4. Adds new classification head:
   ```
   GlobalAveragePooling2D()
       ↓
   Dense(128, ReLU)
       ↓
   Dropout(0.5)
       ↓
   Dense(num_classes, Softmax)
   ```
5. Compiles with Adam optimizer and Sparse Categorical Crossentropy loss

### 6. 🧪 Trained All Three Models

| Model | Pre-trained On | Parameters (approx.) | Training Epochs |
|-------|---------------|----------------------|-----------------|
| **VGG16** | ImageNet | ~138M (frozen) | 1 |
| **ResNet50** | ImageNet | ~25M (frozen) | 1 |
| **EfficientNetB0** | ImageNet | ~5.3M (frozen) | 1 |

*Note: Models were trained for 1 epoch due to compute constraints in the Colab environment*

---

## 🛠️ Libraries Used

```python
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import tensorflow as tf, kagglehub
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from sklearn.metrics import confusion_matrix, classification_report
```

---

## 🏛️ Architecture Comparison

| Feature | VGG16 | ResNet50 | EfficientNetB0 |
|---------|-------|----------|----------------|
| Year | 2014 | 2015 | 2019 |
| Depth | 16 layers | 50 layers | Compound scaled |
| Key Innovation | Simple stacked convs | Skip connections | Compound scaling |
| Size | Large (~528MB) | Medium (~100MB) | Small (~29MB) |
| Speed | Slow | Medium | Fast |

---

## 💡 Key Takeaways

- **Transfer Learning** is a game-changer. By using weights pre-trained on 1.2 million ImageNet images, the model already "knows" how to detect edges, textures, and shapes before seeing a single soyabean image.
- **Freezing the base model** during the first phase is important — it prevents destroying the pre-trained knowledge before the new head has a chance to learn.
- Using `image_dataset_from_directory()` is much cleaner than `ImageDataGenerator` for modern TensorFlow workflows.
- **EfficientNetB0** is the most efficient of the three — great accuracy with far fewer parameters, making it ideal for deployment.
- **ResNet50's skip connections** solve the vanishing gradient problem, allowing it to be deep (50 layers) without degrading performance.
- Running on CPU (no GPU available in this session) makes training slow — this is where cloud GPUs and TPUs become essential for real-world projects.
