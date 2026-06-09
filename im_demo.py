import clip  # see: https://github.com/openai/CLIP
import numpy as np
import torch
from PIL import Image

from splytters.metrics import mean_dist

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

filenames = [
    "test_data/barcodes/barcode_1.jpg",
    "test_data/barcodes/barcode_2.jpg",
    "test_data/barcodes/barcode_3.jpg",
    "test_data/barcodes/barcode_4.jpg",
    "test_data/barcodes/barcode_5.jpg",
    "test_data/barcodes/barcode_6.jpg",
    "test_data/barcodes/barcode_7.jpg",
    "test_data/barcodes/barcode_8.jpg",
]

images = [Image.open(fname) for fname in filenames]

embeddings = []
for im in images:
    im = preprocess(im).unsqueeze(0).to(device)
    embeddings.append(model.encode_image(im).detach().numpy())

embeddings = np.concatenate(embeddings, axis=0)

div = mean_dist(embeddings)

print(div)
