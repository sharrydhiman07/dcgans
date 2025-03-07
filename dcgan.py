import torch
print("CUDA Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

"""
Optimized DCGAN for NVIDIA A100 GPU
---------------------------------------
- Uses PyTorch with Mixed Precision (AMP) for faster training
- Saves the trained model after each epoch
- Trains on the CIFAR-10 dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# ✅ Set device to CUDA (A100 detected)
device = torch.device("cuda")
print(f"Using device: {device} - {torch.cuda.get_device_name(0)}")

# ✅ Enable Mixed Precision (AMP) for faster training
from torch.amp import autocast, GradScaler
scaler = GradScaler()  # ✅ Correct


# ✅ Hyperparameters optimized for A100
lr = 0.0002
batch_size = 256  # ⬆️ Increased batch size (A100 has 40GB VRAM)
image_size = 32
z_dim = 100
num_epochs = 120
features_gen = 64
features_disc = 64
beta1 = 0.5

# ✅ Create directory to save models
os.makedirs("saved_models", exist_ok=True)

# ✅ Data transformation pipeline
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)  # Normalize images to [-1,1] range
])

# ✅ Load CIFAR-10 dataset
dataset = datasets.CIFAR10(root="data", train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d * 4, 1, 4, 1, 0, bias=False)
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, features_g * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g, channels_img, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

# ✅ Initialize models
gen = Generator(z_dim, 3, features_gen).to(device)
disc = Discriminator(3, features_disc).to(device)

# ✅ Loss and Optimizers
criterion = nn.BCEWithLogitsLoss()
optimizer_gen = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_disc = optim.Adam(disc.parameters(), lr=lr, betas=(beta1, 0.999))

# ✅ Noise for visualization
fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

# ✅ Training Loop with AMP
print("Starting Training...")
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(tqdm(dataloader)):
        real = real.to(device, non_blocking=True)
        noise = torch.randn(batch_size, z_dim, 1, 1, device=device)

        with autocast(device_type="cuda"):  # ✅ Mixed Precision
            fake = gen(noise)

            # Train Discriminator
            optimizer_disc.zero_grad()
            disc_real = disc(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))

            disc_fake = disc(fake.detach()).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

            lossD = (lossD_real + lossD_fake) / 2

        scaler.scale(lossD).backward()
        scaler.step(optimizer_disc)
        scaler.update()

        # Train Generator
        optimizer_gen.zero_grad()
        with autocast(device_type="cuda"):  # ✅ Mixed Precision
            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))

        scaler.scale(lossG).backward()
        scaler.step(optimizer_gen)
        scaler.update()

    # ✅ Save generated images
    with torch.no_grad():
        fake_images = gen(fixed_noise).detach().cpu()
        vutils.save_image(fake_images, f"generated_epoch_{epoch}.png", normalize=True)

    # ✅ Save model checkpoints
    torch.save(gen.state_dict(), f"saved_models/dcgan_generator_epoch_{epoch}.pth")
    torch.save(disc.state_dict(), f"saved_models/dcgan_discriminator_epoch_{epoch}.pth")
    print(f"Model saved at epoch {epoch}")

print("Training Complete!")


import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

# ✅ Set device to GPU (or CPU if CUDA is unavailable)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Define Generator class (must match training architecture)
class Generator(torch.nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(z_dim, features_g * 4, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(features_g * 4),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(features_g * 2),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(features_g),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(features_g, channels_img, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

# ✅ Hyperparameters (must match training)
z_dim = 100
features_gen = 64
channels_img = 3

# ✅ Load trained generator
gen = Generator(z_dim, channels_img, features_gen).to(device)
gen.load_state_dict(torch.load("saved_models/dcgan_generator_epoch_119.pth", map_location=device))  # Load latest checkpoint
gen.eval()  # Set to evaluation mode

print("✅ Model loaded successfully!")


# ✅ Generate Noise (Latent Vector)
num_samples = 16  # Change to generate more images
noise = torch.randn(num_samples, z_dim, 1, 1, device=device)

# ✅ Generate Fake Images
with torch.no_grad():
    fake_images = gen(noise).cpu()

# ✅ Display Generated Images
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(vutils.make_grid(fake_images, padding=2, normalize=True), (1, 2, 0)))
plt.show()
