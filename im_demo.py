import clip # see: https://github.com/openai/CLIP
import torch
import numpy as np
from PIL import Image

from metrics import mean_dist

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

filenames = [
    "data/barcode_1.jpg",
    "data/barcode_2.jpg",
    "data/barcode_3.jpg",
    "data/barcode_4.jpg",
    "data/barcode_5.jpg",
    "data/barcode_6.jpg",
    "data/barcode_7.jpg",
    "data/barcode_8.jpg",
]

images = [Image.open(fname) for fname in filenames]

embeddings = []
for im in images:
    im = preprocess(im).unsqueeze(0).to(device)
    embeddings.append(model.encode_image(im).detach().numpy())

embeddings = np.concatenate(embeddings, axis=0)

div = mean_dist(embeddings)

print(div)
