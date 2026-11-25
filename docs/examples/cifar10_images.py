"""Image classification on a CIFAR-10 sample with CLIP image embeddings."""

import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from splytters import (
    cluster_split,
    distance_adversarial_split,
    per_class_split,
    random_split,
    sorted_stratified_split,
)
from splytters.embedders import CLIPImageEmbedder
from splytters.sorters import mean_brightness

# 1. Download CIFAR-10 and take 300 training images per class.
ds = load_dataset("uoft-cs/cifar10", split="train").shuffle(seed=0)
labels = np.asarray(ds["label"])
keep = np.concatenate([np.flatnonzero(labels == c)[:300] for c in range(10)])
images = [ds[int(i)]["img"].convert("RGB") for i in keep]
y = labels[keep]
print(f"{len(images)} images, {len(np.unique(y))} classes")

# 2. Format: embed the images in batches with CLIP.
embedder = CLIPImageEmbedder()
X = np.vstack([embedder.embed(images[i:i + 64]) for i in range(0, len(images), 64)])

# 3. Split, including a sorter-driven split: the darkest images of each class go to train.
splits = {
    "random_split": random_split(X, train_size=0.7, random_state=0),
    "cluster_split (per class)": per_class_split(
        cluster_split, X, y, train_size=0.7, random_state=0),
    "distance_adversarial_split (per class)": per_class_split(
        distance_adversarial_split, X, y, train_size=0.7, random_state=0),
    "mean_brightness (dark to train)": sorted_stratified_split(
        mean_brightness(images), y, train_size=0.7),
}

# 4. Train a classifier on each training split and evaluate it on the test split.
print(f"{'split':40s} {'accuracy':>8s}")
for name, (train_idx, test_idx) in splits.items():
    clf = LogisticRegression(max_iter=2000).fit(X[train_idx], y[train_idx])
    print(f"{name:40s} {accuracy_score(y[test_idx], clf.predict(X[test_idx])):8.3f}")
