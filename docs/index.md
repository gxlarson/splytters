# splytters

**Clustering-based data splits for model robustness testing** — adversarial,
overlap, and distribution-balanced train-test splitting, with a
scikit-learn-aligned API.

Standard random splits assume train and test are drawn from the same
distribution. `splytters` lets you deliberately control the train/test
relationship in embedding space — to make evaluation *harder* (adversarial),
*easier* (overlap, for sanity checks), or *fair* (distribution-balanced) — so
you can probe how a model behaves under covariate shift instead of assuming it
away.

## Install

```bash
pip install splytters
```

The core install depends only on `numpy`, `scipy`, and `scikit-learn`. Optional
extras add modality sorters and embedders — see the [Guide](guide.md).

## Quickstart

```python
import numpy as np
from splytters import cluster_split, split_report

# Any (n_samples, n_features) embedding matrix.
X = np.random.default_rng(0).normal(size=(500, 16))

# An adversarial split: whole clusters go to train OR test (no leakage).
train_idx, test_idx = cluster_split(X, train_size=0.7)

# Quantify how hard/similar the split is (centroid/coverage/leakage + MMD,
# energy distance, Wasserstein, KS).
report = split_report(X, train_idx, test_idx)
print(report)
```

Drop-in for `sklearn.model_selection.train_test_split`:

```python
from splytters import adversarial_train_test_split

X_train, X_test, y_train, y_test = adversarial_train_test_split(
    X, y, train_size=0.7
)
```

...or as a cross-validator anywhere scikit-learn expects `cv=`:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from splytters import SplytterSplit

cv = SplytterSplit(embeddings=X, n_splits=5)
scores = cross_validate(LogisticRegression(), X, y, cv=cv)
```

## The three split families

| Family | Goal | Use it to |
| --- | --- | --- |
| **Adversarial** | *minimize* train↔test similarity | stress-test under covariate shift |
| **Overlap** | *maximize* train↔test similarity | sanity-check / establish an upper bound |
| **Balanced** | *match* train↔test distributions | fair, like-for-like evaluation |

<div class="grid" markdown>
![Adversarial splits](adv.png)
![Overlap splits](overlap.png)
![Balanced splits](bal.png)
</div>

Each row is a 2D distribution (unimodal, moons, spirals, rings, ...) and each
column a splitter; blue = train, orange = test.

Read on in the [Guide](guide.md), or jump to the full
[API reference](api.md).
