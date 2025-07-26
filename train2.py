import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tqdm import tqdm

# --------------------- CONFIG ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

latent_dim = 100
num_classes = 26  # A-Z
image_size = 28
channels = 1
batch_size = 64
epochs = 6000
sample_interval = 20

# --------------------- MODEL: Generator ---------------------
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_shape):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, latent_dim)

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, noise, labels):
        gen_input = noise * self.label_emb(labels)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img

# --------------------- MODEL: Discriminator ---------------------
class Discriminator(nn.Module):
    def __init__(self, num_classes, img_shape):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, int(torch.prod(torch.tensor(img_shape))))

        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        flat_img = img.view(img.size(0), -1)
        d_in = flat_img * self.label_embedding(labels)
        validity = self.model(d_in)
        return validity

# --------------------- PREP ---------------------
img_shape = (channels, image_size, image_size)
os.makedirs("samples", exist_ok=True)

generator = Generator(latent_dim, num_classes, img_shape).to(device)
discriminator = Discriminator(num_classes, img_shape).to(device)

adversarial_loss = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# --------------------- DATA ---------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = ImageFolder(root="dataset2", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --------------------- TRAIN ---------------------
for epoch in range(epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        batch_size = imgs.size(0)

        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        real_imgs = imgs.to(device)
        labels = labels.to(device)

        # ----------------- Train Generator -----------------
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim, device=device)
        gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)

        gen_imgs = generator(z, gen_labels)
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ----------------- Train Discriminator -----------------
        optimizer_D.zero_grad()

        real_pred = discriminator(real_imgs, labels)
        real_loss = adversarial_loss(real_pred, valid)

        z = torch.randn(batch_size, latent_dim, device=device)
        gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
        gen_imgs = generator(z, gen_labels).detach()

        fake_pred = discriminator(gen_imgs, gen_labels)
        fake_loss = adversarial_loss(fake_pred, fake)

        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

    print(f"[Epoch {epoch+1}/{epochs}]  D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")

    if (epoch + 1) % sample_interval == 0:
        with torch.no_grad():
            sample_z = torch.randn(25, latent_dim, device=device)
            sample_labels = torch.randint(0, num_classes, (25,), device=device)
            sample_imgs = generator(sample_z, sample_labels)
            save_image(sample_imgs, f"samples/epoch_{epoch+1}.png", nrow=5, normalize=True)
