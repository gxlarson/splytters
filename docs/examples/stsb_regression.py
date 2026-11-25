"""Regression on STS-B: sentence-pair similarity scores from 0 to 5."""

import numpy as np
from datasets import load_dataset
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from splytters import (
    cluster_split,
    distance_adversarial_split,
    min_cut_split,
    random_split,
    sorted_stratified_split,
)
from splytters.embedders import TextEmbedder

# 1. Download STS-B (the GLUE training split).
ds = load_dataset("nyu-mll/glue", "stsb", split="train")
y = np.asarray(ds["label"], dtype=float)
print(f"{len(y)} sentence pairs, scores from {y.min():.0f} to {y.max():.0f}")

# 2. Format: one vector per pair, [u, v, |u - v|], from the two sentence embeddings.
embedder = TextEmbedder("all-MiniLM-L6-v2")
u, v = embedder.embed(ds["sentence1"]), embedder.embed(ds["sentence2"])
X = np.hstack([u, v, np.abs(u - v)])

# 3. Split. Regression has no classes, so the splitters run on the whole set.
#    The target-sorted split puts the lowest scores in train and the highest in
#    test, by treating all pairs as one class in sorted_stratified_split.
splits = {
    "random_split": random_split(X, train_size=0.7, random_state=0),
    "cluster_split": cluster_split(X, train_size=0.7, random_state=0),
    "distance_adversarial_split": distance_adversarial_split(X, train_size=0.7),
    "min_cut_split": min_cut_split(X, train_size=0.7),
    "target-sorted (low scores to train)": sorted_stratified_split(
        np.argsort(y, kind="stable"), np.zeros(len(y), dtype=int), train_size=0.7),
}

# 4. Train a ridge regressor and report the Spearman correlation on the test split.
print(f"{'split':38s} {'spearman':>8s} {'mean score train/test':>22s}")
for name, (train_idx, test_idx) in splits.items():
    model = Ridge(alpha=1.0).fit(X[train_idx], y[train_idx])
    rho = spearmanr(y[test_idx], model.predict(X[test_idx])).correlation
    means = f"{y[train_idx].mean():.2f} / {y[test_idx].mean():.2f}"
    print(f"{name:38s} {rho:8.3f} {means:>22s}")
