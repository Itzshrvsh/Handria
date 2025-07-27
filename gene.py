import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import random
import sys
import math

# ------------------- SETTINGS -------------------
class S:
    latent_dim = 100
    num_classes = 62
    image_size = 28
    char_scale = 0.7
    spacing_letter = 2
    spacing_word = 16
    line_spacing = 10
    canvas_size = (1000, 1600)
    model_path = 'letter_generator_optimized.pth'
    save_path = 'generated_text_output.png'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    font_path = "Camiro.ttf"
    font_size = 18
    text_color = (0, 0, 255)
    canvas_color = (255, 255, 255)
    upper_scale = 0.9   # Bigger for uppercase letters
    lower_scale = 0.7   # Smaller for lowercase letters
    digit_scale = 0.75  # Optional, if you want digits to differ

    # Human Variation Preset
    human_mode = "neat"  # 'neat', 'rushed', 'drunk'

    # Auto-adjust by mode
    if human_mode == 'neat':
        messiness = 0.1
        contrast_level = 9
        brightness_range = (0.97, 1.02)
        rotation_range = (-2, 2)
        letter_wiggle_range = 1
        line_slant_max_deg = 1
        random_scale_range = (0.98, 1.02)
        jitter_strength = 0.5
        overdraw_chance = 0.0
    elif human_mode == 'rushed':
        messiness = 0.5
        contrast_level = 8
        brightness_range = (0.9, 1.1)
        rotation_range = (-4, 4)
        letter_wiggle_range = 2
        line_slant_max_deg = 3
        random_scale_range = (0.93, 1.07)
        jitter_strength = 1
        overdraw_chance = 0.2
    elif human_mode == 'drunk':
        messiness = 1.0
        contrast_level = 9
        brightness_range = (0.85, 1.15)
        rotation_range = (-6, 6)
        letter_wiggle_range = 4
        line_slant_max_deg = 6
        random_scale_range = (0.85, 1.15)
        jitter_strength = 2
        overdraw_chance = 0.4

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

    if char.isupper():
        scale = S.upper_scale
    elif char.islower():
        scale = S.lower_scale
    elif char.isdigit():
        scale = S.digit_scale
    else:
        scale = S.char_scale  # fallback

    scaled_size = int(S.image_size * scale)
    fake_img = fake_img.resize((scaled_size, scaled_size), Image.LANCZOS)

    # Enhance image with variable contrast/brightness
    contrast = ImageEnhance.Contrast(fake_img).enhance(
        random.uniform(S.contrast_level - 2, S.contrast_level + 1))
    brightness_factor = random.uniform(*S.brightness_range)
    fake_img = ImageEnhance.Brightness(contrast).enhance(brightness_factor)

    return fake_img.convert("L")

# ------------------- HUMAN EFFECTS -------------------
def apply_human_effects(img, line_index=0):
    angle = random.uniform(*S.rotation_range) * S.messiness
    img = img.rotate(angle, fillcolor=255)
    scale = random.uniform(*S.random_scale_range)
    w, h = img.size
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    slant_angle = random.uniform(-S.line_slant_max_deg, S.line_slant_max_deg)
    img = img.transform(
        img.size,
        Image.AFFINE,
        (1, math.tan(math.radians(slant_angle)), 0, 0, 1, 0),
        resample=Image.BICUBIC,
        fillcolor=255
    )
    return img

# ------------------- PUNCTUATION -------------------
def draw_punctuation(draw, x, y, char):
    font_size = int(S.font_size * S.char_scale)
    font = ImageFont.truetype(S.font_path, font_size)
    draw.text((x, y), char, font=font, fill=S.text_color)
    return x + font_size // 2

# ------------------- USER INPUT -------------------
def get_multiline_input(prompt="📝 Enter text (type '<<<' on a new line to finish):"):
    print(prompt)
    lines = []
    while True:
        line = sys.stdin.readline()
        if line.strip() == "<<<":
            break
        lines.append(line.rstrip('\n'))
    return "\n".join(lines)

# ------------------- MAIN -------------------
def main():
    generator = Generator(S.latent_dim, S.num_classes).to(S.device)
    generator.load_state_dict(torch.load(S.model_path, map_location=S.device))
    generator.eval()

    text = get_multiline_input()

    canvas = Image.new("RGB", S.canvas_size, color=S.canvas_color)
    draw = ImageDraw.Draw(canvas)
    x, y = 20, 20
    word_buffer = []
    drunk_level = 0

    for idx, char in enumerate(text):
        if char in [' ', '\n']:
            for char_img in word_buffer:
                jitter_x = int(random.uniform(-S.letter_wiggle_range, S.letter_wiggle_range) * S.messiness)
                jitter_y = int(random.uniform(-S.jitter_strength, S.jitter_strength) * S.messiness)
                canvas.paste(char_img, (x + jitter_x, y + jitter_y), mask=char_img.convert("L"))
                x += char_img.size[0] + S.spacing_letter
            word_buffer = []

            if char == ' ':
                x += S.spacing_word
                drunk_level += 1
            elif char == '\n':
                x = 20
                y += S.image_size + S.line_spacing
            continue

        elif char in char_to_label:
            base_img = generate_char_image(generator, char)
            if base_img is not None:
                base_img = apply_human_effects(base_img, y // (S.image_size + S.line_spacing))

                if random.random() < S.overdraw_chance:
                    base_img2 = apply_human_effects(base_img.copy())
                    base_img2 = base_img2.resize(base_img.size, Image.LANCZOS)
                    base_img = Image.blend(base_img, base_img2, alpha=0.5)

                rgb_img = Image.new("RGB", base_img.size, S.text_color)
                mask = base_img.point(lambda p: 255 - p)
                colored = Image.composite(rgb_img, Image.new("RGB", base_img.size, S.canvas_color), mask)
                word_buffer.append(colored)

        else:
            x = draw_punctuation(draw, x, y + 2, char)

        if x + S.image_size > S.canvas_size[0]:
            x = 20
            y += S.image_size + S.line_spacing

    for char_img in word_buffer:
        jitter_x = int(random.uniform(-S.letter_wiggle_range, S.letter_wiggle_range) * S.messiness)
        jitter_y = int(random.uniform(-S.jitter_strength, S.jitter_strength) * S.messiness)
        canvas.paste(char_img, (x + jitter_x, y + jitter_y), mask=char_img.convert("L"))
        x += char_img.size[0] + S.spacing_letter

    canvas.save(S.save_path)
    print("✅ Image saved as:", S.save_path)

if __name__ == "__main__":
    main()