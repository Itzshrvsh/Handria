import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# Hyperparams
latent_dim = 100
num_classes = 62  # A-Z
image_size = 28
batch_size = 64
epochs = 20000
lr = 0.0002

# Data
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize for Tanh [-1, 1]
])

dataset = ImageFolder(root='dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, image_size * image_size),
            nn.Tanh()
        )

    def forward(self, z, labels):
        x = z + self.label_emb(labels)
        img = self.model(x)
        return img.view(img.size(0), 1, image_size, image_size)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, image_size * image_size)
        self.model = nn.Sequential(
            nn.Linear(image_size * image_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        flat_img = img.view(img.size(0), -1)
        x = flat_img * self.label_emb(labels)
        return self.model(x)

# Init models
generator = Generator(latent_dim, num_classes).to(device)
discriminator = Discriminator(num_classes).to(device)

# Optimizers & Loss
adversarial_loss = nn.BCELoss()
opt_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Training Loop
for epoch in range(epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        batch_size = imgs.size(0)
        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        real_imgs = imgs.to(device)
        labels = labels.to(device)

        # ------------------
        #  Train Generator
        # ------------------
        z = torch.randn(batch_size, latent_dim).to(device)
        gen_imgs = generator(z, labels)
        g_loss = adversarial_loss(discriminator(gen_imgs, labels), valid)

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), labels), fake)
        d_loss = (real_loss + fake_loss) / 2

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

    print(f"[Epoch {epoch+1}/{epochs}]  D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")

    # Save sample output
    if (epoch + 1) % 20 == 0:
        sample_z = torch.randn(26, latent_dim).to(device)
        sample_labels = torch.arange(0, 26).to(device)
        sample_imgs = generator(sample_z, sample_labels)
        save_image(sample_imgs, f"samples/epoch_{epoch+1}.png", normalize=True)

# Save final Generator model
torch.save(generator.state_dict(), "letter_generator_optimized.pth")
print("✅ Optimized model saved as 'letter_generator_optimized.pth'")
