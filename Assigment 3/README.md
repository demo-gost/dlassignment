# ⚙️ Assignment 3 — Forward & Backward Propagation from Scratch

## 📌 Objective

In this lab, I went deeper than just calling `model.fit()`. I **manually implemented** the training loop using TensorFlow's `GradientTape` API to understand exactly how forward propagation, backpropagation, and weight updates work under the hood. I also experimented with different **learning rates** and **epoch counts** to understand their effect on model performance.

---

## 📂 Files

| File | Description |
|------|-------------|
| `lab3.ipynb` | Main notebook with manual training loop and hyperparameter experiments |

---

## 📖 What I Did — Step by Step

### 1. 📦 Loaded & Explored the Wine Dataset
- Used the **same Wine dataset** as Assignment 2 (178 samples, 13 features, 3 classes)
- Checked shape, feature names, class names, and class distribution:
  - class_1: 71 samples
  - class_0: 59 samples
  - class_2: 48 samples
- Split: **80% train (142)** / **20% test (36)**
- Set random seeds (`np.random.seed(42)`, `tf.random.set_seed(42)`) for reproducibility
- **TensorFlow Version used:** 2.20.0

### 2. ⚖️ Standardized the Data
- Applied `StandardScaler` — fit on training data, transform on both sets

### 3. 🏗️ Defined the Model Architecture
```
Input (13 features)
    ↓
Dense(16, ReLU)
    ↓
Dense(8, ReLU)
    ↓
Dense(3, Softmax)
```
- **Optimizer:** Adam with configurable learning rate

### 4. 🔬 Implemented Forward Propagation Manually
- Wrote a `forward_propagation(model, X)` function that calls the model in inference mode (`training=False`)
- Tested on 5 sample training points and confirmed the output shape was `(5, 3)` (5 samples, 3 class probabilities each)

### 5. 🔄 Implemented Backpropagation Manually with `GradientTape`
- Wrote a `train_one_step(model, X_batch, y_batch, optimizer)` function:
  1. **Forward pass:** Run prediction inside `tf.GradientTape()` context and compute loss
  2. **Backward pass:** Call `tape.gradient()` to compute gradients of loss w.r.t. all trainable weights
  3. **Weight update:** Apply gradients using `optimizer.apply_gradients()`

### 6. 🏋️ Wrote a Full Custom Training Loop
- Implemented `train_model(learning_rate, epochs, batch_size=32)` which:
  - Shuffles data every epoch using random permutation
  - Processes data in **mini-batches**
  - Tracks epoch loss and training accuracy
  - Prints progress every 10 epochs

**Training Results (lr=0.001, 100 epochs):**

| Epoch | Loss | Training Accuracy |
|-------|------|-----------------|
| 1 | 5.9763 | 30.28% |
| 20 | 3.2602 | 91.55% |
| 50 | 0.4199 | 99.30% |
| 100 | 0.0870 | **100%** |

**Test Accuracy: 94.44%**

### 7. 📊 Full Model Evaluation
- **Classification Report:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|---------|
| class_0 | 0.92 | 1.00 | 0.96 |
| class_1 | 0.93 | 0.93 | 0.93 |
| class_2 | 1.00 | 0.90 | 0.95 |
| **Overall** | **0.95** | **0.94** | **0.94** |

- Generated a **Confusion Matrix** heatmap
- Plotted **Training Loss vs Epoch** and **Training Accuracy vs Epoch** graphs

### 8. 🔬 Hyperparameter Experiments

#### Learning Rate Comparison
Tested 4 learning rates (0.0001, 0.001, 0.01, 0.1) — **best result: lr=0.01 → 97.22% test accuracy**

#### Epoch Count Comparison
Tested epoch counts of [10, 25, 50, 100, 200] — **best result: 100 epochs → 97.22% test accuracy**

---

## 🛠️ Libraries Used

```python
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

---

## 💡 Key Takeaways

- Implementing backpropagation manually with `GradientTape` is incredibly educational — you see exactly what `model.fit()` is doing internally!
- **Mini-batch training** with shuffling helps the model converge faster and more stably than using the whole dataset at once.
- **Learning rate** has a huge impact — too small (0.0001) and the model barely learns; too large (0.1) and it becomes unstable. `0.01` hit the sweet spot.
- **100 epochs** was the optimal number — training longer didn't improve test accuracy but could lead to overfitting.
- The gap between 100% training accuracy and 94% test accuracy indicates some overfitting, which would be addressed by adding regularization (as done in Assignment 2).
