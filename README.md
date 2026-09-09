# 👋 My Deep Learning Journey

Welcome to my deep learning repository! This is where I've been keeping all the Jupyter Notebooks for my assignments. It's basically a log of my progress—starting from the absolute basics of neural networks and working all the way up to some pretty cool image classification using transfer learning. 

Everything here is built using Python, TensorFlow, and Keras.

## 🗂️ What's Inside?

Here's a quick tour of what I've been working on:

*   **[Lab 1: Dipping My Toes into Neural Networks](lab1.ipynb)**
    *   **What it is:** My very first time building a neural network with TensorFlow and Keras.
    *   **What I learned:** I got the hang of data preprocessing and built a simple Multi-Layer Perceptron (MLP). I used it to classify different types of Iris flowers based on the classic dataset!
    
*   **[Lab 2: Stopping the Overfitting Monster](lab2.ipynb)**
    *   **What it is:** A deep dive into why models overfit and how to stop them.
    *   **What I learned:** I worked with the Wine dataset and experimented with L1/L2 regularization. I also set up `EarlyStopping` callbacks so the model knows when to quit training before it just memorizes the training data.

*   **[Lab 3: Going Deeper with Classification](lab3.ipynb)**
    *   **What it is:** Pushing the neural network architecture a bit further.
    *   **What I learned:** More multi-class classification on the Wine dataset, but this time I spent more time evaluating the model using accuracy scores, confusion matrices, and detailed classification reports to really understand where it was succeeding and failing.

*   **[Lab 6: Seeing the World with CNNs](lab6.ipynb)**
    *   **What it is:** My introduction to Convolutional Neural Networks (CNNs) and Computer Vision!
    *   **What I learned:** Instead of tabular data, I started feeding images into the model. I used `ImageDataGenerator` to load and augment a dataset I pulled from Kaggle, and built my very own CNN from scratch.

*   **[Lab 7: Standing on the Shoulders of Giants (Transfer Learning)](lab7.ipynb)**
    *   **What it is:** Why train from scratch when you can use models that are already super smart?
    *   **What I learned:** I tackled advanced image classification using transfer learning. I fine-tuned massive pre-trained models like VGG16, ResNet50, and EfficientNetB0, utilizing `image_dataset_from_directory` to handle the data smoothly.

## 🛠️ The Tools I Used

*   **Python** (The bread and butter)
*   **TensorFlow & Keras** (The heavy lifters for building the models)
*   **Scikit-Learn** (Super handy for grabbing datasets and evaluation metrics)
*   **Pandas & NumPy** (For wrangling all that data)
*   **Matplotlib & Seaborn** (For making sense of the numbers through cool graphs)

## 🚀 Want to try it out?

If you want to poke around and run the code yourself, it's pretty straightforward:

1. Make sure you have Python installed.
2. Install the packages I used:
   ```bash
   pip install tensorflow scikit-learn pandas numpy matplotlib seaborn kagglehub
   ```
3. Fire up Jupyter Notebook (or JupyterLab) and open any of the `.ipynb` files:
   ```bash
   jupyter notebook
   ```

Thanks for stopping by and checking out my work!
