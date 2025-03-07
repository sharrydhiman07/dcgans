# DCGAN on CIFAR-10

## Overview
This project trains a Deep Convolutional Generative Adversarial Network (DCGAN) on the CIFAR-10 dataset.

## Dataset Preprocessing
- CIFAR-10 images are resized to `32x32` pixels.
- Images are normalized to `[-1,1]` for stable GAN training.

## Training the Model
### **Step 1: Install Dependencies**

### **Step 2: Run Training**
- Trains DCGAN for 50 epoch.
- Saves generated images at the end of training.

## Expected Outputs
After training, the model will generate **fake CIFAR-10 images** resembling objects like cars, cats, and airplanes.

## Deployment Requirements
- Python 3.8+
- PyTorch 2.0+
- NVIDIA GPU (Recommended)
