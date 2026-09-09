# ⚔️ Assignment 2 — Fighting Overfitting with Regularization & Early Stopping

## 📌 Objective

This lab was all about one of the biggest challenges in deep learning: **overfitting**. I explored three techniques to prevent a model from memorizing the training data — L1 regularization, L2 regularization, and Early Stopping — and compared their performance on the Wine dataset.

---

## 📂 Files

| File | Description |
|------|-------------|
| `lab2.ipynb` | Main notebook containing all code, experiments, and results |

---

## 📖 What I Did — Step by Step

### 1. 📦 Loaded the Wine Dataset
- Used `sklearn.datasets.load_wine` to load the Wine dataset
- **178 samples**, **13 chemical features**, **3 classes** (wine varieties: class_0, class_1, class_2)
- Converted to a Pandas DataFrame and verified — **no missing values, no duplicates**
- Split into **80% training (142 samples)** and **20% testing (36 samples)**, stratified by class

### 2. ⚖️ Standardized the Features
- Applied `StandardScaler` to normalize all 13 chemical features
- Fitted the scaler on training data only and then transformed both train and test sets (to avoid data leakage)

### 3. 🏗️ Built a Reusable Model Factory
- Created a `create_model(regularizer=None)` function that builds the same architecture every time, but allows plugging in different regularization strategies

**Model Architecture:**

| Layer | Details |
|-------|---------|
| Dense (Hidden 1) | 64 neurons, ReLU, optional kernel regularizer, input shape = (13,) |
| Dense (Hidden 2) | 32 neurons, ReLU, optional kernel regularizer |
| Dense (Output) | 3 neurons, Softmax |

- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Epochs:** 100, **Batch Size:** 16

### 4. 🧪 Ran 4 Experiments

| # | Model | Technique | Test Accuracy |
|---|-------|-----------|--------------|
| 1 | Basic MLP | No regularization | **94.44%** |
| 2 | MLP + L1 | L1 regularization (λ=0.001) | **97.22%** |
| 3 | MLP + L2 | L2 regularization (λ=0.001) | **97.22%** |
| 4 | MLP + Early Stopping | Stop training when val_loss stops improving (patience=10) | **97.22%** |

### 5. 📊 Evaluated the Best Model (Early Stopping)
- Generated a **Confusion Matrix** visualized as a heatmap:
  ```
  [[12  0  0]
   [ 0 14  0]
   [ 0  1  9]]
  ```
- Generated a full **Classification Report:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|---------|
| class_0 | 1.00 | 1.00 | 1.00 |
| class_1 | 0.93 | 1.00 | 0.97 |
| class_2 | 1.00 | 0.90 | 0.95 |
| **Overall** | **0.97** | **0.97** | **0.97** |

### 6. 📈 Compared Model Accuracies with a Bar Chart
- Plotted all 4 models side-by-side to visually confirm that regularization (any flavor) consistently outperforms a bare model

---

## 🛠️ Libraries Used

```python
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

---

## 💡 Key Takeaways

- A **bare model** without any regularization achieved 94.44%, which is decent — but it's more likely to overfit on unseen data.
- Both **L1 and L2 regularization** pushed accuracy up to 97.22% by penalizing large weights and forcing the model to generalize.
- **Early Stopping** also achieved 97.22% with the benefit of automatically picking the best-performing epoch — no manual tuning needed.
- **L1 regularization** tends to push weights to exactly zero (useful for feature selection), while **L2** shrinks them smoothly.
- The Wine dataset turns out to be very well-suited for neural networks — highly accurate results with just two hidden layers!
