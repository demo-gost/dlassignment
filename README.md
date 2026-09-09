# 👋 My Deep Learning Journey

Welcome to my deep learning assignments repository! This is where I've been keeping all the Jupyter Notebooks from my course labs. It's basically a log of my progress — starting from the absolute basics of neural networks all the way up to advanced image classification using state-of-the-art transfer learning architectures.

Everything here is built using **Python, TensorFlow, and Keras**, and all experiments were run on **Google Colab**.

---

## 🗂️ Repository Structure

```
dlassigmnet/
├── Assignment 1/         → TensorFlow setup & first neural network (Iris dataset)
│   ├── lab1.ipynb
│   └── README.md
│
├── Assignment 2/         → Regularization & Early Stopping (Wine dataset)
│   ├── lab2.ipynb
│   └── README.md
│
├── Assigment 3/          → Forward & Backward Propagation from scratch (Wine dataset)
│   ├── lab3.ipynb
│   └── README.md
│
├── Assigment 6/          → CNN for Tomato Image Classification (Kaggle dataset)
│   ├── lab6.ipynb
│   └── README.md
│
└── Assigmnet 7/          → Transfer Learning: VGG16, ResNet50, EfficientNetB0 (Soyabean dataset)
    ├── lab7.ipynb
    └── README.md
```

---

## 📚 Assignment Summaries

### [Assignment 1 — First Neural Network on Iris Dataset](Assignment%201/README.md)
My very first time building a neural network with TensorFlow and Keras. I loaded the classic Iris dataset, preprocessed it (scaling + train-test split), and built a simple MLP (Multi-Layer Perceptron) to classify flower species.

**Key skills:** Data preprocessing, StandardScaler, MLP architecture, Dropout, Softmax

---

### [Assignment 2 — Regularization & Early Stopping](Assignment%202/README.md)
A deep dive into the overfitting problem. I ran 4 experiments on the Wine dataset — a bare model, one with L1 regularization, one with L2 regularization, and one with Early Stopping — and compared their accuracy.

**Key skills:** L1/L2 regularization, EarlyStopping callback, confusion matrix, classification report  
**Best Accuracy Achieved:** 97.22%

---

### [Assignment 3 — Manual Forward & Backpropagation](Assigment%203/README.md)
Instead of just calling `model.fit()`, I implemented the training loop **from scratch** using TensorFlow's `GradientTape`. I also studied the effect of different learning rates and epoch counts on performance.

**Key skills:** `tf.GradientTape`, manual gradient calculation, mini-batch training, hyperparameter tuning  
**Best Test Accuracy:** 97.22% (lr=0.01, 100 epochs)

---

### [Assignment 6 — CNN for Tomato Classification](Assigment%206/README.md)
My first foray into Computer Vision! I built a CNN from scratch to classify tomato images (Ripe, Unripe, Damaged, Old) using a 6,500-image Kaggle dataset with data augmentation.

**Key skills:** CNN architecture, ImageDataGenerator, data augmentation, Conv2D, MaxPooling2D, GlobalAveragePooling2D  
**Dataset:** `enalis/tomatoes-dataset` from Kaggle (6,500 training images)

---

### [Assignment 7 — Transfer Learning](Assigmnet%207/README.md)
The most advanced lab — using pre-trained giants (VGG16, ResNet50, EfficientNetB0) to classify a Soyabean image dataset. I built a reusable transfer learning pipeline that freezes the base model and adds a custom classification head.

**Key skills:** Transfer learning, feature extraction, VGG16, ResNet50, EfficientNetB0, `image_dataset_from_directory`, data prefetching  
**Dataset:** `akansha03mulchandani/soyabean` from Kaggle

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | TensorFlow 2.20+, Keras |
| Data & ML | NumPy, Pandas, Scikit-Learn |
| Visualization | Matplotlib, Seaborn |
| Image/Dataset | KaggleHub, ImageDataGenerator |
| Environment | Google Colab (Python 3.12/3.13) |

---

## 🚀 Getting Started

If you want to run any of the notebooks yourself:

1. Make sure you have Python installed.
2. Install the required dependencies:
   ```bash
   pip install tensorflow scikit-learn pandas numpy matplotlib seaborn kagglehub
   ```
3. Open a notebook using Jupyter:
   ```bash
   jupyter notebook
   ```
   > **Tip:** Labs 6 and 7 require a Kaggle account and API token to download datasets via `kagglehub`. Run them on Google Colab for the best experience.

---

Thanks for checking out my work! 🙌
