import os
import xml.etree.ElementTree as ET
from PIL import Image
import string

# --- Valid characters ---
valid_chars = string.ascii_letters + string.digits + string.punctuation + " "

# --- Character-safe folder names ---
def safe_folder_name(char):
    special_map = {
        " ": "space", "/": "slash", "\\": "backslash", ":": "colon", "*": "asterisk",
        "?": "question", '"': "quote", "<": "lt", ">": "gt", "|": "pipe", ".": "dot",
        "!": "excl", "@": "at", "#": "hash", "$": "dollar", "%": "percent", "^": "caret",
        "&": "and", "(": "lparen", ")": "rparen", "+": "plus", "-": "dash", "=": "equals",
        ",": "comma", ";": "semicolon", "'": "apostrophe", "`": "backtick", "~": "tilde",
        "[": "lbracket", "]": "rbracket", "{": "lbrace", "}": "rbrace", "_": "underscore"
    }
    return special_map.get(char, char)

# --- Setup paths ---
ANNOTATIONS_DIR = "anno"
IMAGES_DIR = "images"
OUTPUT_DIR = "dataset"
def safe_filename(label):
    unsafe = r'\/:*?"<>|'
    return ''.join(safe_folder_name(c) if c in unsafe else c for c in label)
# --- Create folders for all characters ---
folder_map = {}  # map label -> folder path
for char in valid_chars:
    folder_name = safe_folder_name(char)
    folder_path = os.path.join(OUTPUT_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    folder_map[char] = folder_path

# --- Count existing files per character ---
letter_counts = {char: len(os.listdir(folder_map[char])) for char in valid_chars}

# --- Parse and crop from XML ---
for xml_file in os.listdir(ANNOTATIONS_DIR):
    if not xml_file.endswith('.xml'):
        continue

    xml_path = os.path.join(ANNOTATIONS_DIR, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_file = root.find('filename').text
    image_path = os.path.join(IMAGES_DIR, image_file)

    if not os.path.exists(image_path):
        print(f"Image not found: {image_file}")
        continue

    image = Image.open(image_path).convert('L')

    for obj in root.findall('object'):
        label = obj.find('name').text.strip()

        if label not in valid_chars:
            print(f"Skipping unknown label: {label}")
            continue

        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))

        cropped = image.crop((xmin, ymin, xmax, ymax))
        resized = cropped.resize((28, 28))

        folder = safe_folder_name(label)
        safe_name = safe_filename(label)

        letter_counts[label] += 1
        filename = f"{safe_name}_{letter_counts[label]}.png"
        save_path = os.path.join(OUTPUT_DIR, folder, filename)

        resized.save(save_path)
        print(f"Saved: {save_path}")
