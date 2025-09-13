# Self-Supervised Learning with an Autoencoder on MNIST

This repository contains a Jupyter Notebook (`SSL_MNIST_Autoencoder.ipynb`) that demonstrates a complete self-supervised learning (SSL) pipeline. The project uses a Convolutional Autoencoder (CAE) to learn feature representations from the unlabeled MNIST dataset and then transfers this knowledge to a downstream, few-shot classification task.

## Key Concepts Demonstrated
- **Self-Supervised Learning (SSL)** using a reconstruction task.
- **Convolutional Autoencoders (CAE)** for feature extraction.
- **Transfer Learning** from a pre-trained encoder.
- **Few-Shot Classification** with a limited labeled dataset.
- **Baseline Comparison** against a standard supervised model.
- **t-SNE** for embedding visualization.

## Pipeline Overview

The notebook follows these key steps:

1.  **Self-Supervised Pre-training:** A Convolutional Autoencoder is trained on all 60,000 MNIST training images *without using their labels*. The model's objective is to reconstruct the input images, forcing the encoder to learn meaningful, compressed features.

2.  **Feature Extraction:** The trained encoder is then used as a feature extractor to convert the 28x28 images into compact 64-dimensional latent vectors (embeddings).

3.  **Few-Shot Downstream Task:** A small, labeled subset is created (100 samples per class). A simple Logistic Regression classifier is trained on the extracted latent features from this subset.

4.  **Baseline Comparison:** To evaluate the effectiveness of the SSL features, a standard CNN is trained from scratch on the same small, labeled subset.

5.  **Evaluation:** Both models (the SSL-based classifier and the baseline CNN) are evaluated on the full 10,000-image MNIST test set to compare their performance.

## How to Run

1.  Clone the repository:
    ```bash
    git clone [https://github.com/AhmedAlaa612/ssl-intro.git](https://github.com/AhmedAlaa612/ssl-intro.git)
    cd your-repo-name
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Run the Jupyter Notebook:
    ```bash
    jupyter notebook SSL_MNIST_Autoencoder.ipynb
    ```
