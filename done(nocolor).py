import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import random
import sys
# ------------------- SETTINGS -------------------
class S:
    latent_dim = 100
    num_classes = 62
    image_size = 28
    char_scale = 0.7              # 🔁 Set this to control printed letter size
    spacing_letter = 2           # Adjust if increasing char scale
    spacing_word = 16          # Space between words
    line_spacing = 10            # More space for bigger letters
    canvas_size = (1000, 1600)   # Widen canvas for large letters
    model_path = 'letter_generator_optimized.pth'
    save_path = 'generated_text_output.png'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
    contrast_level = 9
    brightness_range = (0.95, 1.05)
    rotation_range = (-3, 3)
    font_path = "Camiro.ttf"
    font_size = 18               # Base font size (will scale)
    text_color = (0, 0, 255)  # Blue color text in RGB. Change as needed
    canvas_color = (255, 255, 255)  # White background



# ------------------- CHAR SET -------------------
char_set = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ")
char_to_label = {char: idx for idx, char in enumerate(char_set)}

# ------------------- MODEL -------------------
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
            nn.Linear(1024, S.image_size * S.image_size),
            nn.Tanh()
        )

    def forward(self, z, labels):
        x = z + self.label_emb(labels)
        img = self.model(x)
        return img.view(img.size(0), 1, S.image_size, S.image_size)

# ------------------- IMAGE GENERATION -------------------
def generate_char_image(generator, char):
    if char not in char_to_label:
        return None
    label = char_to_label[char]
    noise = torch.randn(1, S.latent_dim).to(S.device)
    label_tensor = torch.tensor([label], dtype=torch.long).to(S.device)
    with torch.no_grad():
        fake_img = generator(noise, label_tensor)
    fake_img = fake_img.squeeze().cpu()
    fake_img = transforms.ToPILImage()(fake_img)
    scaled_size = int(S.image_size * S.char_scale)
    fake_img = fake_img.resize((scaled_size, scaled_size), Image.LANCZOS)


    # Add contrast
    fake_img = ImageEnhance.Contrast(fake_img).enhance(S.contrast_level)

    # Brightness variation
    brightness_factor = random.uniform(*S.brightness_range)
    fake_img = ImageEnhance.Brightness(fake_img).enhance(brightness_factor)

    # Small rotation
    rotation_angle = random.uniform(*S.rotation_range)
    fake_img = fake_img.rotate(rotation_angle, fillcolor=255)

    return fake_img

# ------------------- DRAW PUNCTUATION -------------------
def draw_punctuation(draw, x, y, char):
    font_size = int(S.font_size * S.char_scale)
    font = ImageFont.truetype(S.font_path, font_size)
    draw.text((x, y), char, font=font, fill=0)
    return x + font_size // 2

# ------------------- MAIN -------------------
def get_multiline_input(prompt="📝 Enter text (type '<<<' on a new line to finish):"):
    print(prompt)
    lines = []
    while True:
        line = sys.stdin.readline()
        if line.strip() == "<<<":
            break
        lines.append(line.rstrip('\n'))
    return "\n".join(lines)

def main():
    generator = Generator(S.latent_dim, S.num_classes).to(S.device)
    generator.load_state_dict(torch.load(S.model_path, map_location=S.device))
    generator.eval()
    canvas = Image.new("RGB", S.canvas_size, color=S.canvas_color)

    text = get_multiline_input()

    canvas = Image.new("L", S.canvas_size, color=255)
    draw = ImageDraw.Draw(canvas)
    scaled_size = int(S.image_size * S.char_scale)
    x, y = 20, 20
    for char in text:
        if char == '\n':
            x = 20
            y += S.image_size + S.line_spacing
            continue
        elif char == ' ':
            x += S.spacing_word
            continue
        elif char in char_to_label:
            char_img = generate_char_image(generator, char)
            if char_img:
                canvas.paste(char_img, (x, y))
                x += scaled_size + S.spacing_letter

        else:
            x = draw_punctuation(draw, x, y + 2, char)

        if x + S.image_size > S.canvas_size[0]:
            x = 20
            y += S.image_size + S.line_spacing

    canvas.save(S.save_path)
    print("✅ Image saved as:", S.save_path)

if __name__ == "__main__":
    main()
