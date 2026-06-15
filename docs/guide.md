# Guide

`splytters` operates on **embeddings** — any `(n_samples, n_features)` array
(NumPy, a list of vectors, or a torch tensor). Every splitter takes that matrix
plus a `train_size` (a fraction in `(0, 1)` **or** an absolute integer count)
and returns a pair of integer index arrays `(train_indices, test_indices)`.

```python
train_idx, test_idx = cluster_split(X, train_size=0.7, random_state=42)
```

## Discovering what's available

To see every splitter or sorter without reading the source:

```python
import splytters

splytters.list_splitters()                 # flat list of every splitter
splytters.list_splitters(by_family=True)   # {"adversarial": [...], "overlap": [...], ...}

splytters.list_sorters()                   # flat list of every sorter
splytters.list_sorters(by_modality=True)   # {"embedding": [...], "text": [...], ...}

splytters.list_embedders()                  # ["TextEmbedder", "CLIPTextEmbedder", ...]
```

`list_sorters` and `list_embedders` pull in no optional dependency — they
return just the names, so the modality sorters and embedder models stay lazily
imported until you actually use one.

## Writing a custom splitter

Anything matching the `splytters.Splitter` type — a callable taking an
embeddings array (plus `train_size` / `random_state`) and returning a
`(train_indices, test_indices)` pair of integer arrays — works anywhere a
splitter is accepted (`SplytterSplit`, `splytter_train_test_split`, the interop
helpers):

```python
from splytters import Splitter, splytter_train_test_split

def my_split(embeddings, *, train_size=0.7, random_state=42):
    ...
    return train_idx, test_idx

X_train, X_test = splytter_train_test_split(X, splitter=my_split)
```

## Choosing a split family

### Adversarial — minimize train↔test similarity

Make evaluation harder by pushing train and test apart in embedding space, so
the test set probes generalization under covariate shift.

- `cluster_split` — assign whole clusters to one side (prevents cluster leakage)
- `centroid_adversarial_split`, `distance_adversarial_split`,
  `density_adversarial_split`, `outlier_adversarial_split`
- `min_cut_split`, `normalized_cut_split` — graph-cut formulations
- `wasserstein_adversarial_split` — Wasserstein nearest-neighbor split
  (Søgaard et al., EACL 2021)

### Overlap — maximize train↔test similarity

Make evaluation easier: useful as a sanity check or an optimistic upper bound.

- `cluster_leak_split`, `neighbor_coverage_split`, `centroid_matched_split`,
  `stratified_similarity_split`, `nearest_neighbor_split`,
  `duplicate_spread_split`, `max_coverage_split`

### Balanced — match train↔test distributions

Fair, like-for-like evaluation where both sides look statistically similar.

- `distribution_matched_split`, `moment_matched_split`,
  `histogram_matched_split`, `stratified_random_split`,
  `density_balanced_split`, `mmd_minimized_split`

A plain `random_split` baseline is also provided.

### Curriculum — split along a sorted order

`sorted_stratified_split` takes an explicit ordering (typically a
[sorter](#sorters) ranking) plus class labels, and within each class assigns the
first `train_size` fraction to train and the rest to test — a "train on easy,
test on hard" curriculum split:

```python
from splytters.sorters import readability_score
from splytters import sorted_stratified_split

ranking = readability_score(texts)        # [(index, score), ...] easy -> hard
train_idx, test_idx = sorted_stratified_split(ranking, y, train_size=0.7)
# largest_first=True flips it to "train on hard, test on easy"
```

Unlike the embedding-based families above, this one is driven by the ordering
you give it rather than by the embeddings, so it isn't listed by
`list_splitters()`.

## Scoring a split

`split_report` quantifies how adversarial/overlapping/balanced a split actually
is — centroid distance, coverage, cluster-leakage, plus MMD, energy distance,
mean Wasserstein, and mean KS (and optional label-distribution shift if you pass
`y`). `compare_splitters` runs several splitters and tabulates the results.

```python
from splytters import split_report

report = split_report(X, train_idx, test_idx, y=labels)
```

## scikit-learn integration

- **`SplytterSplit`** — a `BaseCrossValidator` usable as `cv=` in
  `cross_validate`, `GridSearchCV`, etc.
- **`splytter_train_test_split`** and the family wrappers
  **`adversarial_/overlap_/balanced_train_test_split`** mirror
  `sklearn.model_selection.train_test_split`: for inputs `a, b` they return
  `[a_train, a_test, b_train, b_test]`.

## Framework interop

Heavy dependencies are imported lazily, so these helpers don't weigh down the
core install:

- **`split_dataframe`** (pandas) — split a `DataFrame`, optionally embedding a
  subset of columns
- **`to_torch_subsets`** (torch) — wrap a split as `torch.utils.data.Subset`s
- **`split_dataset`** (HuggingFace `datasets`) — split a `Dataset`

Splitters also accept torch tensors directly.

## Sorters

The `splytters.sorters` subpackage ranks samples by interpretable
difficulty/quality metrics — handy for curriculum-style splits ("train on easy,
test on hard") or just inspecting data:

```python
from splytters.sorters import distance_to_mean

ranked = distance_to_mean(embeddings)  # most typical → most atypical
```

Each modality lives behind an optional extra and is imported lazily:
`embedding_sorters` (core), `text_sorters` (`pip install "splytters[text]"`),
`image_sorters` (`[image]`), `audio_sorters` (`[audio]`),
`tabular_sorters` (`[tabular]`).

## Optional extras

| Extra | Adds |
| --- | --- |
| `text`, `image`, `audio`, `tabular` | modality sorters |
| `embedders` | `TextEmbedder`, CLIP, OpenAI embedders |
| `ann` | approximate-nearest-neighbor backend for large-n splits |
| `viz` | matplotlib |
| `all` | everything above |

See the [API reference](api.md) for full signatures and parameters.
