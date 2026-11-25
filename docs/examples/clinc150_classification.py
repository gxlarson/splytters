"""Text classification on CLINC-150: download, embed, split, train and evaluate."""

import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from splytters import (
    cluster_leak_split,
    cluster_split,
    distance_adversarial_split,
    minority_grow_split,
    per_class_split,
    random_split,
    sorted_stratified_split,
    split_report,
)
from splytters.embedders import TextEmbedder
from splytters.sorters import tokens_length

# 1. Download CLINC-150 and keep the 150 in-scope intents.
ds = load_dataset("clinc/clinc_oos", "plus", split="train")
oos = ds.features["intent"].names.index("oos")
ds = ds.filter(lambda row: row["intent"] != oos)
texts, y = ds["text"], np.asarray(ds["intent"])
print(f"{len(texts)} queries, {len(np.unique(y))} intents")

# 2. Format: splitters take an (n_samples, dim) array of embeddings.
X = TextEmbedder("all-MiniLM-L6-v2").embed(texts)

# 3. Split. per_class_split runs a splitter within each class, so that every
#    intent keeps 70% of its queries in train.
splits = {
    "random_split": random_split(X, train_size=0.7, random_state=0),
    "cluster_leak_split (per class)": per_class_split(
        cluster_leak_split, X, y, train_size=0.7, random_state=0),
    "cluster_split (per class)": per_class_split(
        cluster_split, X, y, train_size=0.7, random_state=0),
    "distance_adversarial_split (per class)": per_class_split(
        distance_adversarial_split, X, y, train_size=0.7, random_state=0),
    "minority_grow_split": minority_grow_split(
        X, y, train_size=0.7, n_clusters=150, stratify="per_class", random_state=0),
    # A sorter-driven split: the shortest queries of each intent go to train.
    "tokens_length (short to train)": sorted_stratified_split(
        tokens_length(list(texts)), y, train_size=0.7),
}

# 4. Train a classifier on each training split and evaluate it on the test split.
print(f"{'split':40s} {'accuracy':>8s} {'c2st_auc':>8s}")
for name, (train_idx, test_idx) in splits.items():
    clf = LogisticRegression(max_iter=1000).fit(X[train_idx], y[train_idx])
    acc = accuracy_score(y[test_idx], clf.predict(X[test_idx]))
    c2st = split_report(X, train_idx, test_idx, max_samples=1000)["c2st_auc"]
    print(f"{name:40s} {acc:8.3f} {c2st:8.3f}")
