import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from gan_model import Generator, Discriminator
import signal
import sys

# Configuration
latent_dim = 100
channels = 3
img_size = 32
batch_size = 64
n_epochs = 200
checkpoint_interval = 10
model_save_path = "saved_models"
os.makedirs(model_save_path, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
generator = Generator(latent_dim, channels).to(device)
discriminator = Discriminator(channels).to(device)

# Load existing model if available
def load_checkpoint():
    if os.path.exists(os.path.join(model_save_path, "generator.pth")):
        generator.load_state_dict(torch.load(os.path.join(model_save_path, "generator.pth")))
        discriminator.load_state_dict(torch.load(os.path.join(model_save_path, "discriminator.pth")))
        print("Loaded existing model checkpoint")
        return True
    return False

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
adversarial_loss = nn.BCELoss()

# Configure data loader
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = datasets.ImageFolder(root="images", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Training history
g_losses = []
d_losses = []

def save_checkpoint(epoch):
    torch.save(generator.state_dict(), os.path.join(model_save_path, "generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join(model_save_path, "discriminator.pth"))
    print(f"Saved checkpoint at epoch {epoch}")

def plot_metrics(epoch):
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.savefig(os.path.join(model_save_path, f'loss_plot_epoch_{epoch}.png'))
    plt.close()

def signal_handler(sig, frame):
    print("\nTraining interrupted! Saving model...")
    save_checkpoint(current_epoch)
    plot_metrics(current_epoch)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Training loop
current_epoch = 0
if load_checkpoint():
    current_epoch = int(input("Enter the epoch number to continue from: "))

print("Starting training...")
for epoch in range(current_epoch, n_epochs):
    current_epoch = epoch
    g_loss_epoch = []
    d_loss_epoch = []
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for i, (imgs, _) in enumerate(progress_bar):
        batch_size = imgs.shape[0]
        real_imgs = imgs.to(device)

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim).to(device)
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), torch.ones(batch_size, 1).to(device))
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs), torch.ones(batch_size, 1).to(device))
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), torch.zeros(batch_size, 1).to(device))
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        g_loss_epoch.append(g_loss.item())
        d_loss_epoch.append(d_loss.item())
        
        progress_bar.set_postfix({
            'G_Loss': np.mean(g_loss_epoch),
            'D_Loss': np.mean(d_loss_epoch)
        })

    g_losses.append(np.mean(g_loss_epoch))
    d_losses.append(np.mean(d_loss_epoch))

    if epoch % checkpoint_interval == 0:
        save_checkpoint(epoch)
        plot_metrics(epoch)
        
        # Save sample images
        os.makedirs("generated_images", exist_ok=True)
        save_image(gen_imgs.data[:25], f"generated_images/epoch_{epoch}.png", nrow=5, normalize=True)

print("Training completed!")
save_checkpoint(n_epochs)
plot_metrics(n_epochs) 