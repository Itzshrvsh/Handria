import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageEnhance

# ------------------- SETTINGS -------------------
class S:
    latent_dim = 100
    num_classes = 68
    image_size = 28
    spacing_letter = 4
    spacing_word = 12
    line_spacing = 16
    canvas_size = (1080, 1920)
    margin = 30
    contrast_factor = 9.0
    model_path = "letter_generator_optimized.pth"
    text_color = (0, 0, 0)
    background = (255, 255, 255)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- MODEL -------------------
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, S.image_size * S.image_size),
            nn.Tanh()
        )

    def forward(self, z, labels):
        embedded = self.label_emb(labels)
        x = z * embedded
        img = self.model(x)
        return img.view(img.size(0), 1, S.image_size, S.image_size)

# Load trained generator
model = Generator(S.latent_dim, S.num_classes).to(device)
model.load_state_dict(torch.load(S.model_path, map_location=device))
model.eval()

# ------------------- UTILS -------------------
def apply_color(img_gray):
    """Convert grayscale to binary black-on-white using contrast and thresholding."""
    enhancer = ImageEnhance.Contrast(img_gray)
    img = enhancer.enhance(S.contrast_factor)

    colored = Image.new("RGB", img.size, S.background)
    px = img.load()
    out_px = colored.load()

    for x in range(img.width):
        for y in range(img.height):
            if px[x, y] < 128:
                out_px[x, y] = S.text_color

    return colored

def generate_char_image(label_id):
    """Generate single character image from label ID."""
    z = torch.randn(1, S.latent_dim).to(device)
    label_tensor = torch.tensor([label_id]).to(device)

    with torch.no_grad():
        generated = model(z, label_tensor)

    img_tensor = (generated.squeeze().cpu() + 1) * 127.5
    img_tensor = img_tensor.clamp(0, 255).byte()
    img_gray = transforms.ToPILImage(mode="L")(img_tensor)

    return apply_color(img_gray)

# ------------------- GENERATE SEQUENCE -------------------
def generate_text_image(label_sequence):
    canvas_w, canvas_h = S.canvas_size
    canvas = Image.new("RGB", (canvas_w, canvas_h), S.background)

    x_cursor = S.margin
    y_cursor = S.margin

    for label_id in label_sequence:
        if label_id == -1:
            # Handle space
            x_cursor += S.spacing_word
            continue

        char_img = generate_char_image(label_id)

        if x_cursor + char_img.width > canvas_w - S.margin:
            x_cursor = S.margin
            y_cursor += S.image_size + S.line_spacing

        if y_cursor + S.image_size > canvas_h - S.margin:
            print("⚠️ Canvas full, text cut off.")
            break

        canvas.paste(char_img, (x_cursor, y_cursor))
        x_cursor += char_img.width + S.spacing_letter

    return canvas

# ------------------- MAIN -------------------
if __name__ == "__main__":
    # Example: label sequence (A, B, C, space, D, E, F)
    label_sequence = [0, 1, 2, -1, 3, 4, 5, 6, -1, 7, 8, 9]  # Adjust for your label mappings

    os.makedirs("outputs", exist_ok=True)
    out_img = generate_text_image(label_sequence)
    out_img.save("outputs/generated_text.png")
    print("✅ Saved: outputs/generated_text.png")
