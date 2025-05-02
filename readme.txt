# 🧠 DCGAN – Deep Convolutional GAN for Image Generation

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate realistic images from random noise. It uses PyTorch to define both the generator and discriminator, and trains the model on a dataset of images.

---

## 🌟 Features

- 🧑‍🎨 Generates new images using a trained generator
- 🕹️ Discriminator and generator trained adversarially
- 🎯 Supports custom image size and dataset
- 🖼️ Real-time training output saved to disk

---

## 🛠️ Tech Stack

- Python 3.x
- PyTorch
- torchvision
- NumPy
- Matplotlib / PIL

---

## 🚀 Getting Started

### 1. Clone the repo

git clone https://github.com/sharrydhiman07/dcgans.git
cd dcgans

2. Install dependencies

pip install torch torchvision numpy matplotlib pillow

3. Run the training script

python dcgan.py
🧾 Model Summary
Generator: Uses transposed convolutions to upscale random noise into 64x64 images.

Discriminator: A CNN-based binary classifier that tries to distinguish real from fake images.

Loss: Binary Cross Entropy (BCE)

Optimizer: Adam (lr=0.0002, beta1=0.5)




📦 Folder Structure

dcgans/
├── dcgan.py               # Main training script
├── dataset/               # Training images (e.g., CelebA, MNIST, custom)
├── output/                # Generated images per epoch
├── README.md
