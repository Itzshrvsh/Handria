import os
import numpy as np
from PIL import Image

IMG_SIZE = 28
DATASET_DIR = 'dataset/'
LABELS = sorted(os.listdir(DATASET_DIR))  # A to Z
label_to_idx = {label: idx for idx, label in enumerate(LABELS)}

def load_data():
    images = []
    labels = []
    for label in LABELS:
        folder = os.path.join(DATASET_DIR, label)
        for img_file in os.listdir(folder):
            img = Image.open(os.path.join(folder, img_file)).convert('L').resize((IMG_SIZE, IMG_SIZE))
            img = np.array(img) / 255.0  # normalize
            images.append(img)
            labels.append(label_to_idx[label])
    return np.array(images), np.array(labels)
