import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from tqdm import tqdm

# ------------------ SETTINGS ------------------
latent_dim = 100
image_size = 64
batch_size = 64
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ MODEL ------------------
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
        img_flat = img.view(img.size(0), -1)
        label_flat = self.label_emb(labels)
        x = img_flat * label_flat
        return self.model(x)

# ------------------ DATASET LOADING ------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # [-1, 1]
])

# Map classes A-Z, a-z, 0–9 => total 62 classes
def get_class_index(c):
    if 'A' <= c <= 'Z':
        return ord(c) - ord('A')  # 0-25
    elif 'a' <= c <= 'z':
        return 26 + ord(c) - ord('a')  # 26-51
    elif '0' <= c <= '9':
        return 52 + ord(c) - ord('0')  # 52–61
    else:
        return -1

# Override ImageFolder to remap classes to 0–61
class CustomFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        classes = sorted(os.listdir(directory))
        class_to_idx = {cls_name: get_class_index(cls_name) for cls_name in classes if get_class_index(cls_name) >= 0}
        return classes, class_to_idx

dataset1 = CustomFolder("dataset1", transform=transform)
dataset2 = CustomFolder("dataset2", transform=transform)
full_dataset = ConcatDataset([dataset1, dataset2])
dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

# ------------------ INIT MODELS ------------------
num_classes = 62
generator = Generator(latent_dim, num_classes).to(device)
discriminator = Discriminator(num_classes).to(device)

# ------------------ LOSS + OPTIM ------------------
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# ------------------ TRAINING ------------------
os.makedirs("generated_samples", exist_ok=True)

for epoch in range(epochs):
    loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)
    for i, (imgs, labels) in enumerate(loop):
        imgs, labels = imgs.to(device), labels.to(device)
        batch_size = imgs.size(0)

        # Valid / Fake labels
        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        # ------------------ Train Generator ------------------
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim).to(device)
        gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
        gen_imgs = generator(z, gen_labels)
        g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)
        g_loss.backward()
        optimizer_G.step()

        # ------------------ Train Discriminator ------------------
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(imgs, labels), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        loop.set_description(f"[Epoch {epoch+1}/{epochs}]")
        loop.set_postfix(G_loss=g_loss.item(), D_loss=d_loss.item())

    if (epoch + 1) % 10 == 0:
        sample = generator(torch.randn(16, latent_dim).to(device),
                           torch.randint(0, num_classes, (16,), device=device))
        save_image(sample.data, f"generated_samples/epoch_{epoch+1}.png", nrow=4, normalize=True)

# ------------------ SAVE MODEL ------------------
torch.save(generator.state_dict(), "letter_generator_optimized.pth")
print("✅ Saved model: letter_generator_optimized.pth")
