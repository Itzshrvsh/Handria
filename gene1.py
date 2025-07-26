import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import torch.nn as nn
import os

# ------------------ SETTINGS ------------------
class Settings:
    latent_dim = 100
    num_classes = 26
    image_size = 28
    spacing_letter = 6
    spacing_word = 30
    line_spacing = 40
    contrast_factor = 9.0
    canvas_size = (1080, 1920)  # 🛠️ WIDTH x HEIGHT — auto wrap on small width
    margin = 50
    model_path = "letter_generator_optimized.pth"
    text_color = (0, 0, 255)  # 🔵 Blue

S = Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ MODEL ------------------
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
            nn.Linear(1024, S.image_size * S.image_size),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = noise + self.label_emb(labels)
        img = self.model(gen_input)
        return img.view(img.size(0), 1, S.image_size, S.image_size)

model = Generator(S.latent_dim, S.num_classes).to(device)
model.load_state_dict(torch.load(S.model_path, map_location=device))
model.eval()

# ------------------ UTILS ------------------
def apply_color_to_letter(img_gray):
    enhancer = ImageEnhance.Contrast(img_gray)
    img = enhancer.enhance(S.contrast_factor)

    colored = Image.new("RGB", img.size, (255, 255, 255))
    px = img.load()
    out_px = colored.load()

    for x in range(img.width):
        for y in range(img.height):
            if px[x, y] < 128:
                out_px[x, y] = S.text_color

    return colored

# ------------------ GENERATE LETTER ------------------
def generate_letter_image(letter):
    z = torch.randn(1, S.latent_dim).to(device)

    if letter == " ":
        return Image.new("RGB", (S.spacing_word, S.image_size), (255, 255, 255))

    elif letter == ".":
        img = Image.new("RGB", (S.image_size, S.image_size), (255, 255, 255))
        dot = Image.new("RGB", (4, 4), S.text_color)
        img.paste(dot, (S.image_size // 2 - 2, S.image_size - 10))
        return img

    elif not letter.isalpha():
        return Image.new("RGB", (S.image_size // 2, S.image_size), (255, 255, 255))

    label_index = ord(letter.upper()) - ord('A')
    label_tensor = torch.tensor([label_index]).to(device)

    with torch.no_grad():
        generated = model(z, label_tensor)

    img_tensor = (generated.squeeze().cpu() + 1) * 127.5
    img_tensor = img_tensor.clamp(0, 255).byte()
    img_gray = transforms.ToPILImage(mode="L")(img_tensor)

    return apply_color_to_letter(img_gray)

# ------------------ GENERATE WORD IMAGE ------------------
def generate_word_image(word):
    letters = [generate_letter_image(c) for c in word]
    total_width = sum(img.width for img in letters) + (len(letters) - 1) * S.spacing_letter
    word_img = Image.new("RGB", (total_width, S.image_size), (255, 255, 255))

    x_offset = 0
    for img in letters:
        y_pad = (S.image_size - img.height) // 2
        word_img.paste(img, (x_offset, y_pad))
        x_offset += img.width + S.spacing_letter

    return word_img

# ------------------ GENERATE PARAGRAPH IMAGE ------------------
def generate_paragraph_image(text):
    canvas_width, canvas_height = S.canvas_size
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))

    x_cursor = S.margin
    y_cursor = S.margin

    words = text.strip().split()

    for word in words:
        word_img = generate_word_image(word)
        word_width = word_img.width

        # Line wrap
        if x_cursor + word_width > canvas_width - S.margin:
            x_cursor = S.margin
            y_cursor += S.image_size + S.line_spacing

        # Canvas overflow
        if y_cursor + S.image_size > canvas_height - S.margin:
            print("⚠️ Text truncated: too long for canvas")
            break

        canvas.paste(word_img, (x_cursor, y_cursor))
        x_cursor += word_width + S.spacing_word

    return canvas

# ------------------ MAIN ------------------
if __name__ == "__main__":
    sentence = "This is a dynamic paragraph generator using GAN-generated letters. The lines wrap automatically when width is limited."
    os.makedirs("outputs", exist_ok=True)
    final_img = generate_paragraph_image(sentence)
    filename = sentence[:30].replace(" ", "_").lower()
    final_img.save(f"outputs/output.png")
    print(f"✅ Saved as outputs/{filename}_wrapped.png")
