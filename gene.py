import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import torch.nn as nn
import os

# --------------------- SETTINGS ---------------------
latent_dim = 100
num_classes = 26
image_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------- MODEL ---------------------
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

# Load model
generator = Generator(latent_dim, num_classes).to(device)
generator.load_state_dict(torch.load("letter_generator_optimized.pth", map_location=device))
generator.eval()

# ------------------ GENERATE SINGLE LETTER IMAGE ------------------
def generate_letter_image(letter):
    if letter == " ":
        return Image.new("L", (40, image_size), 255)
    elif letter == ".":
        img = Image.new("L", (30, image_size), 255)
        dot = Image.new("L", (10, 10), 0)
        img.paste(dot, (10, image_size - 20))
        return img
    elif not letter.isalpha():
        return Image.new("L", (30, image_size), 255)

    z = torch.randn(1, latent_dim).to(device)
    label_index = ord(letter.upper()) - ord('A')
    label_tensor = torch.tensor([label_index]).to(device)

    with torch.no_grad():
        gen_img = generator(z, label_tensor)

    # Rescale from [-1, 1] to [0, 255]
    img_tensor = (gen_img.squeeze().cpu() + 1) * 127.5
    img_tensor = img_tensor.clamp(0, 255).byte()
    img = transforms.ToPILImage()(img_tensor)

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)

    return img

# ------------------ GENERATE FULL SENTENCE IMAGE ------------------
def generate_word_image(word, spacing=6):
    letters = [generate_letter_image(c) for c in word]
    total_width = sum(img.width for img in letters) + (len(letters) - 1) * spacing
    final_img = Image.new('L', (total_width, image_size), color=255)

    x_offset = 0
    for img in letters:
        y_pad = (image_size - img.height) // 2
        final_img.paste(img, (x_offset, y_pad))
        x_offset += img.width + spacing

    return final_img

# ------------------ MAIN ------------------
if __name__ == "__main__":
    sentence = "The quick brown fox jumps over the lazy dog"
    output = generate_word_image(sentence)
    os.makedirs("outputs", exist_ok=True)
    filename = sentence.replace(" ", "_").lower()
    path = f"outputs/{filename}_generated.png"
    output.save(path)
    print(f"✅ Image saved: {path}")
