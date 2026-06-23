# splytters

[![PyPI](https://img.shields.io/pypi/v/splytters.svg)](https://pypi.org/project/splytters/)
[![CI](https://github.com/gxlarson/splytters/actions/workflows/ci.yml/badge.svg)](https://github.com/gxlarson/splytters/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/gxlarson/splytters/branch/main/graph/badge.svg)](https://codecov.io/gh/gxlarson/splytters)
[![Docs](https://readthedocs.org/projects/splytters/badge/?version=latest)](https://splytters.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/gxlarson/splytters/blob/main/LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/splytters.svg)](https://pypi.org/project/splytters/)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Train/test splitting algorithms for dataset partitioning — beyond random splits.

```bash
pip install splytters
```

A random split tells you how a model performs on data that looks just like its training set. Often that's not the question you're asking. This library provides splitters with three different objectives:

| Objective | Package module | What it does | Use it for |
|---|---|---|---|
| **Adversarial** | `splytters.adversarial` | Minimize train/test similarity | Hard evaluation — measure generalization to unfamiliar data |
| **Overlap** | `splytters.overlap` | Maximize train/test similarity | Easy evaluation — sanity checks, debugging, upper-bound estimates |
| **Balanced** | `splytters.balanced` | Match train/test distributions | Fair evaluation — avoid accidental distribution shift |

All splitters operate on embeddings (any `(n_samples, dim)` array — numpy, lists,
pandas, or torch tensors) and return integer index arrays, so they work with any
data you can embed: text, images, audio, tabular rows.

## Installation

```bash
pip install splytters
```

The core install (numpy, scipy, scikit-learn) covers **every splitter**. Optional
extras add modality-specific dependencies for the `sorters` and built-in embedders:

```bash
pip install "splytters[text]"       # text sorters (pysbd, transformers, wordfreq, ...)
pip install "splytters[image]"      # image sorters (pillow)
pip install "splytters[audio]"      # audio sorters (librosa)
pip install "splytters[tabular]"    # tabular sorters (pandas)
pip install "splytters[embedders]"  # built-in embedders (sentence-transformers, ...)
pip install "splytters[all]"        # all of the above
```

Requires Python 3.10+ (tested on 3.10–3.14).

## Quickstart

```python
import numpy as np
from splytters import cluster_split

embeddings = np.random.rand(500, 384)  # your embeddings here

train_idx, test_idx = cluster_split(embeddings, train_size=0.7)
```

Every splitter follows the same scikit-learn-style interface:

```python
train_indices, test_indices = some_split(
    embeddings,        # (n_samples, embedding_dim) array-like
    train_size=0.7,    # fraction in (0, 1) OR an absolute count
    random_state=42,
)
```

A more realistic example with text:

```python
from sentence_transformers import SentenceTransformer
from splytters import centroid_adversarial_split

texts = [...]  # your dataset
embeddings = SentenceTransformer("all-MiniLM-L6-v2").encode(texts)

train_idx, test_idx = centroid_adversarial_split(embeddings, train_size=0.7)
train = [texts[i] for i in train_idx]
test = [texts[i] for i in test_idx]
```

## Works with scikit-learn, pandas, PyTorch & HF datasets

Drop a splitter into any scikit-learn workflow — as a cross-validator or a
`train_test_split` replacement:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from splytters import SplytterSplit, adversarial_train_test_split, cluster_split

# 1) as a CV object (single hard split) in cross_validate / GridSearchCV
cv = SplytterSplit(cluster_split, embeddings=X, n_clusters=10)
cross_validate(LogisticRegression(), X, y, cv=cv)

# 2) as a train_test_split drop-in (splits every array the same way)
X_tr, X_te, y_tr, y_te = adversarial_train_test_split(X, y, embeddings=X)
```

Native helpers for the rest of the ecosystem (heavy deps imported lazily):

```python
from splytters import split_dataframe, to_torch_subsets, split_dataset

train_df, test_df   = split_dataframe(df, embeddings)              # pandas
train_ds, test_ds   = to_torch_subsets(torch_dataset, embeddings)  # PyTorch
dsdict              = split_dataset(hf_dataset, embeddings)        # → DatasetDict
```

## How hard is my split? — `split_report`

```python
from splytters import split_report, compare_splitters, random_split, cluster_split

compare_splitters(embeddings, {"random": random_split, "adversarial": cluster_split})
# {'random': {...}, 'adversarial': {'mmd_rbf': ..., 'energy_distance': ...,
#                                   'wasserstein_mean': ..., 'coverage': ..., ...}}
```

`split_report` quantifies how adversarial/overlapping/balanced a split actually
is (centroid & nearest-train distance, coverage, cluster leakage, MMD, energy
distance, mean 1-D Wasserstein/KS, and optional label-distribution shift).

## Available splitters

**Adversarial** (minimize train/test similarity):
`cluster_split`, `centroid_adversarial_split`, `distance_adversarial_split`, `density_adversarial_split`, `outlier_adversarial_split`, `min_cut_split`, `normalized_cut_split`, `wasserstein_adversarial_split`, `mmd_maximized_split`, `minority_split`, `class_boundary_split`

![Adversarial splitters on 2D distributions](https://raw.githubusercontent.com/gxlarson/splytters/main/docs/adv.png)

**Overlap** (maximize train/test similarity):
`cluster_leak_split`, `neighbor_coverage_split`, `centroid_matched_split`, `stratified_similarity_split`, `nearest_neighbor_split`, `duplicate_spread_split`, `max_coverage_split`

![Overlap splitters on 2D distributions](https://raw.githubusercontent.com/gxlarson/splytters/main/docs/overlap.png)

**Balanced** (match train/test distributions):
`distribution_matched_split`, `moment_matched_split`, `histogram_matched_split`, `stratified_random_split`, `density_balanced_split`, `mmd_minimized_split`

![Balanced splitters on 2D distributions](https://raw.githubusercontent.com/gxlarson/splytters/main/docs/bal.png)

**Supervised** (label-aware — these take class labels `y`):
`class_boundary_split`, `minority_split`, `stratified_random_split`, `sorted_stratified_split`, `cluster_split(strategy="subset_sum")`

![Supervised splitters on labeled 2D distributions](https://raw.githubusercontent.com/gxlarson/splytters/main/docs/supervised.png)

A plain `random_split` baseline and utilities (`compute_pairwise_distances`, `compute_split_similarity`, `cluster_embeddings`, ...) are also exported from `splytters`.

The unsupervised figures are generated by `demos/visualize_splits.py` and the supervised one by `demos/visualize_supervised_splits.py` — each row is a 2D distribution (unimodal, moons, spirals, rings, ...), each column a splitter. Blue = train, orange = test; in the supervised figure marker shape encodes the class. (The supervised figure uses overlapping-class variants of the blob distributions so the label-aware splitters have a contested boundary to work with.)

## Sorters

The companion `splytters.sorters` package ranks samples by interpretable difficulty/quality metrics — useful for curriculum-style splits ("train on easy, test on hard") or just inspecting your data:

- **`embedding_sorters`** — `distance_to_mean`, `distance_to_nearest_neighbor`, `local_density`, `outlier_score`
- **`text_sorters`** — length, readability, perplexity, lexical diversity, vocabulary rarity, sentence count
- **`image_sorters`** — brightness, contrast, color variance, compression ratio, frequency content
- **`audio_sorters`** — loudness, spectral features, MFCCs, rhythm, quality metrics
- **`tabular_sorters`** — column- and row-level metrics, categorical handling, multi-column sorting

```python
from splytters.sorters import distance_to_mean

ranked = distance_to_mean(embeddings)  # most typical → most atypical
```

Pair a sorter with `sorted_stratified_split` to turn that ranking into an actual
curriculum split — within each class, the first `train_size` fraction (by the
sort order) becomes train, the rest test:

```python
from splytters.sorters import readability_score
from splytters import sorted_stratified_split

ranking = readability_score(texts)                 # easy → hard
train_idx, test_idx = sorted_stratified_split(ranking, y, train_size=0.7)
# largest_first=True flips it to "train on hard, test on easy"
```

See `demos/demo.py` for sorters on a real dataset (TREC questions), `demos/im_demo.py`
for an image example with CLIP embeddings, and `demos/trec_sorter_experiment.py`
for a TREC curriculum-split benchmark (text sorters + linear SVM) that quantifies
which sorters capture a real difficulty axis.

## Installation

```bash
git clone https://github.com/gxlarson/splytters
cd splytters
pip install -e .             # core splitters (numpy, scipy, scikit-learn)
```

Optional extras, depending on which sorters/demos you use:

```bash
pip install -e ".[text]"     # text sorters (torch, transformers, py-readability-metrics, wordfreq, pysbd)
pip install -e ".[audio]"    # audio sorters (librosa)
pip install -e ".[image]"    # image sorters (Pillow)
pip install -e ".[tabular]"  # tabular sorters (pandas)
pip install -e ".[ann]"      # approximate-NN backend for large datasets (pynndescent)
pip install -e ".[demo]"     # demos (sentence-transformers, datasets, matplotlib, umap-learn)
pip install -e ".[all]"      # everything
```

> **Note:** sorter imports are **lazy per modality** — `import splytters.sorters`
> pulls in no optional dependencies, and each extra is self-sufficient (e.g.
> `[image]` alone powers the image sorters). The core `splytters` install (numpy,
> scipy, scikit-learn) is all the splitters need.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Project layout

```
splytters/                # the installable package
  adversarial.py / overlap.py / balanced.py / utils.py   # splitting algorithms
  sklearn_api.py          # SplytterSplit + *_train_test_split
  interop.py              # pandas / torch / HuggingFace adapters
  report.py               # split_report, compare_splitters
  embedders.py            # text / image embedders (TextEmbedder, CLIP, OpenAI)
  sorters/                # ranking metrics (text, image, audio, embedding, tabular)
tests/                    # pytest suite
test_data/                # sample audio/images/text used by tests
demos/                    # demo.py, im_demo.py (sorter demos: TREC text, CLIP images)
                          # visualize_splits.py + visualize_supervised_splits.py (README figures), visualize_text_splits.py
scripts/                  # generate_test_*.py (regenerate the test_data/ fixtures)
docs/                     # README figures
```
