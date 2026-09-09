# 🧠 Assignment 1 — Setting Up TensorFlow & Building My First Neural Network

## 📌 Objective

The goal of this lab was to get comfortable with TensorFlow and Keras on Google Colab, and to go through the full pipeline of a machine learning project — from loading raw data all the way to training a neural network.

---

## 📂 Files

| File | Description |
|------|-------------|
| `lab1.ipynb` | Main notebook containing all code and outputs |

---

## 📖 What I Did — Step by Step

### 1. 📦 Loaded the Iris Dataset
- Used `sklearn.datasets.load_iris` to load the classic Iris dataset
- The dataset has **150 samples**, **4 features** (sepal length, sepal width, petal length, petal width), and **3 classes** (Setosa, Versicolor, Virginica)
- Converted it into a Pandas DataFrame for easier exploration

### 2. 🔍 Explored the Data
- Checked the shape, data types, missing values, and basic statistics using `df.info()` and `df.describe()`
- **No missing values** found — the dataset was clean!
- Visualized the data using:
  - A **scatter plot** of sepal length vs. petal length
  - **Histograms** of all features to understand their distributions

### 3. ⚖️ Normalized the Features
- Applied `StandardScaler` from Scikit-Learn to scale all features to have a **mean of 0 and standard deviation of 1**
- This is important so that no single feature dominates the neural network training

### 4. ✂️ Split the Data
- Split the data into **80% training (120 samples)** and **20% testing (30 samples)** using `train_test_split`
- Used `stratify=y` to ensure each class was equally represented in both splits

### 5. 🏗️ Built the Neural Network (MLP)
Built a **Sequential model** using Keras with:

| Layer | Details |
|-------|---------|
| Input (Dense) | 16 neurons, ReLU activation, input shape = (4,) |
| Dropout | 20% dropout for regularization |
| Hidden (Dense) | 8 neurons, ReLU activation |
| Output (Dense) | 3 neurons, Softmax activation (one per class) |

### 6. ⚙️ Compiled & Trained the Model
- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Metric:** Accuracy
- Trained for **30 epochs** with validation data

---

## 🛠️ Libraries Used

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

---

## 📊 Key Results

| Detail | Value |
|--------|-------|
| Dataset | Iris (sklearn) |
| Training Samples | 120 |
| Testing Samples | 30 |
| Model | Multi-Layer Perceptron (MLP) |
| Training Epochs | 30 |

---

## 💡 Key Takeaways

- Setting up TensorFlow is straightforward on Google Colab — it comes pre-installed.
- Data preprocessing (scaling + splitting) is arguably more important than the model architecture for clean tabular datasets.
- Even a small MLP (2 hidden layers) works surprisingly well on a simple dataset like Iris.
- Dropout helps prevent overfitting even in small networks.
