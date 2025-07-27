import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import torch.nn as nn
import os

# --------------------- MODEL ---------------------
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),  # 128 x 128
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = noise + self.label_emb(labels)
        img = self.model(gen_input)
        return img.view(img.size(0), 1,28, 28)

# ------------------ SETTINGS ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
num_classes = 26

model = Generator(latent_dim, num_classes).to(device)
model.load_state_dict(torch.load("letter_generator_optimized.pth", map_location=device))
model.eval()

# ------------------ GENERATE LETTER ------------------
def generate_letter_image(letter):
    z = torch.randn(1, latent_dim).to(device)

    if letter == " ":
        return Image.new("L", (28, 28), 255)
    elif letter == ".":
        img = Image.new("L", (20, 28), 255)
        dot = Image.new("L", (10, 10), 0)
        img.paste(dot, (10, 108))  # dot at the bottom
        return img
    elif not letter.isalpha():
        return Image.new("L", (30, 128), 255)

    label_index = ord(letter.upper()) - ord('A')
    label_tensor = torch.tensor([label_index]).to(device)

    with torch.no_grad():
        generated = model(z, label_tensor)

    img_tensor = (generated.squeeze().cpu() + 1) * 127.5
    img_tensor = img_tensor.clamp(0, 255).byte()
    img = transforms.ToPILImage()(img_tensor)

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)

    return img

# ------------------ GENERATE WORD IMAGE ------------------
def generate_word_image(word, spacing=6):
    letters = [generate_letter_image(c) for c in word]
    total_width = sum(img.width for img in letters) + (len(letters) - 1) * spacing
    final_img = Image.new('L', (total_width, 128), color=255)

    x_offset = 0
    for img in letters:
        vertical_padding = (128 - img.height) // 2
        final_img.paste(img, (x_offset, vertical_padding))
        x_offset += img.width + spacing

    return final_img

# ------------------ RUN ------------------
if __name__ == "__main__":
    sentence = "The quick brown fox jumps over the lazy dog"
    output = generate_word_image(sentence)
    os.makedirs("outputs", exist_ok=True)
    filename = sentence.replace(" ", "_").lower()
    output.save(f"outputs/{filename}_generated.png")
    print(f"✅ Generated image saved as outputs/{filename}_generated.png")
