# Examples

These examples go through the whole workflow on public datasets: download a dataset from the
Hugging Face Hub, turn it into embeddings, split it, and train and evaluate a model on each
split. Each example is a script in `docs/examples/`, and the output shown is from running it
on CPU with `pip install "splytters[all]"`.

## Text classification: CLINC-150

CLINC-150 has 150 intents of a task-oriented assistant, such as `balance` or `book_flight`.
The example compares five splitters and one sorter-driven split, all with 70% of each intent
in train.

```python
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
```

Output:

```text
15000 queries, 150 intents
split                                    accuracy c2st_auc
random_split                                0.953    0.525
cluster_leak_split (per class)              0.966    0.472
cluster_split (per class)                   0.874    0.669
distance_adversarial_split (per class)      0.815    0.756
minority_grow_split                         0.556    0.744
tokens_length (short to train)              0.950    0.778
```

`cluster_leak_split`, an overlap splitter, is slightly easier than `random_split`, and the
adversarial splitters are harder: accuracy drops by 7.9 points with `cluster_split`, by 13.8
points with `distance_adversarial_split` and by 39.7 points with `minority_grow_split`.
`c2st_auc` is the AUC of a classifier trained to tell the two splits apart (0.5 means that
they look alike). Splitting by query length produces splits that are easy to tell apart
(0.778) without changing accuracy, so a dissimilar split is not always a harder one.

## Regression: STS-B

STS-B pairs two sentences with a similarity score from 0 to 5. Regression has no classes, so
the splitters run on the whole set, and a target-sorted split is built with
`sorted_stratified_split` by treating every pair as one class.

```python
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
```

Output:

```text
5749 sentence pairs, scores from 0 to 5
split                                  spearman  mean score train/test
random_split                              0.799            2.70 / 2.70
cluster_split                             0.773            2.67 / 2.77
distance_adversarial_split                0.632            3.33 / 1.22
min_cut_split                             0.867            2.91 / 2.22
target-sorted (low scores to train)       0.360            2.00 / 4.33
```

Training on the low scores and testing on the high scores lowers the Spearman correlation from
0.799 to 0.360. `distance_adversarial_split` sends the pairs farthest from the centroid to
test, which here are mostly unrelated pairs with low scores (mean 1.22), and lowers it to
0.632. `min_cut_split` raises it to 0.867, so not every adversarial splitter makes every task
harder.

## Image classification: CIFAR-10

The example embeds 3,000 CIFAR-10 images with CLIP and adds a sorter-driven split on image
brightness.

```python
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
```

Output:

```text
3000 images, 10 classes
split                                    accuracy
random_split                                0.933
cluster_split (per class)                   0.846
distance_adversarial_split (per class)      0.823
mean_brightness (dark to train)             0.944
```

The two adversarial splitters lower accuracy by 8.7 and 11.0 points, whereas training on the
darkest images of each class and testing on the brightest leaves accuracy almost unchanged.

## Reusing a split

Every splitter returns NumPy arrays of indices, so a split can be saved and applied to the raw
data or to any other representation:

```python
import numpy as np

np.savez("split.npz", train=train_idx, test=test_idx)
train_texts = [texts[i] for i in train_idx]
```

See the [sorter gallery](sorter_gallery.md) for what each sorter ranks first and last.
