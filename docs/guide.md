# Guide

`splytters` operates on embeddings, which can be any `(n_samples, n_features)`
array (NumPy, a list of vectors, or a torch tensor). Every splitter takes that
matrix and a `train_size` (a fraction in `(0, 1)` or an absolute integer count)
and returns a pair of integer index arrays `(train_indices, test_indices)`.

```python
train_idx, test_idx = cluster_split(X, train_size=0.7, random_state=42)
```

## Available splitters and sorters

The following functions list every splitter, sorter and embedder:

```python
import splytters

splytters.list_splitters()                 # flat list of every splitter
splytters.list_splitters(by_family=True)   # {"adversarial": [...], "overlap": [...], ...}

splytters.list_sorters()                   # flat list of every sorter
splytters.list_sorters(by_modality=True)   # {"embedding": [...], "text": [...], ...}

splytters.list_embedders()                  # ["TextEmbedder", "CLIPTextEmbedder", ...]
```

`list_sorters` and `list_embedders` return only the names and import no
optional dependency. The modality sorters and embedder models are imported
when you first use one.

## Custom splitters

A custom splitter is any callable that matches the `splytters.Splitter` type: it
takes an embeddings array (with `train_size` and `random_state`) and returns a
`(train_indices, test_indices)` pair of integer arrays. A custom splitter can be
used wherever a splitter is accepted, including `SplytterSplit`,
`splytter_train_test_split` and the interop helpers:

```python
from splytters import Splitter, splytter_train_test_split

def my_split(embeddings, *, train_size=0.7, random_state=42):
    ...
    return train_idx, test_idx

X_train, X_test = splytter_train_test_split(X, splitter=my_split)
```

## Split families

### Adversarial splits

Adversarial splitters minimize the similarity between the training and test
splits in embedding space. The test split then measures generalization under
covariate shift.

- `cluster_split`: assigns each cluster entirely to train or to test, which
  prevents cluster leakage
- `centroid_adversarial_split`, `distance_adversarial_split`,
  `density_adversarial_split`, `outlier_adversarial_split`
- `min_cut_split`, `normalized_cut_split`: graph-cut formulations
- `wasserstein_adversarial_split`: Wasserstein nearest-neighbor split
  (Søgaard et al., EACL 2021)
- `maximin_split`: a farthest-point (k-center) test split that covers the
  embedding space

The label-aware adversarial splitters also take class labels `y`:

- `minority_split`, `minority_grow_split`: seed the test split with the
  per-cluster minority-label ("anti-biased") samples (Reif & Schwartz, 2023).
  `minority_grow_split` then grows the test split to the requested size.
- `class_boundary_split`: assigns the samples close to other classes to test
- `decision_boundary_split`: assigns the samples near the boundary of a
  cross-validated linear model to test

`per_class_split` runs any embedding splitter within each class, so that every
class keeps its share of train and test:

```python
from splytters import distance_adversarial_split, per_class_split

train_idx, test_idx = per_class_split(distance_adversarial_split, X, y, train_size=0.7)
```

### Overlap splits

Overlap splitters maximize the similarity between the training and test splits.
They make evaluation easier, which is useful as a sanity check or as an
optimistic upper bound.

- `cluster_leak_split`, `neighbor_coverage_split`, `centroid_matched_split`,
  `stratified_similarity_split`, `nearest_neighbor_split`,
  `duplicate_spread_split`, `max_coverage_split`
- `central_split`: the mirror of `distance_adversarial_split`. It assigns the
  samples closest to the centroid to test, so the test split contains
  prototypical samples rather than duplicates of training samples. Wrapping it
  in `per_class_split` produces a test split of samples that are typical of
  their class.

### Balanced splits

Balanced splitters match the distributions of the training and test splits, so
that the two splits are statistically similar.

- `distribution_matched_split`, `moment_matched_split`,
  `histogram_matched_split`, `stratified_random_split`,
  `density_balanced_split`, `mmd_minimized_split`

`random_split` is also provided as a baseline.

### Sorted splits

`sorted_stratified_split` takes an explicit ordering (typically the ranking of a
[sorter](#sorters)) and class labels. Within each class, it assigns the first
`train_size` fraction of the ordering to train and the rest to test. With a
difficulty ranking, this trains on easy samples and tests on hard ones:

```python
from splytters.sorters import readability_score
from splytters import sorted_stratified_split

ranking = readability_score(texts)        # [(index, score), ...] easy -> hard
train_idx, test_idx = sorted_stratified_split(ranking, y, train_size=0.7)
# largest_first=True flips it to "train on hard, test on easy"
```

`sorted_stratified_split` does not use embeddings, so `list_splitters()` does
not list it.

For regression, sort by the target and treat every sample as one class. This
trains on the lowest targets and tests on the highest:

```python
import numpy as np
from splytters import sorted_stratified_split

train_idx, test_idx = sorted_stratified_split(
    np.argsort(y_reg, kind="stable"), np.zeros(len(y_reg), dtype=int), train_size=0.7)
```

## Tips

- **Report a random split too.** Show results on `random_split` next to any
  other split, with the splitter, its parameters and the seeds, so that readers
  can see how much the split changed the result.
- **`minority_grow_split` depends on `n_clusters`.** With few clusters and many
  classes, nearly every sample is a per-cluster minority, the seed is much
  larger than the test set, and the subsampled split ends up close to random
  (the function warns when the seed exceeds 1.5x the test size). Setting
  `n_clusters` near the number of classes is a reasonable start.
- **Some splitters ignore the seed.** `distance_adversarial_split` and
  `min_cut_split` are deterministic, so repeating them over seeds does not add
  variation. When the similarity graph of `min_cut_split` is disconnected, it
  assigns whole components to test, smallest first.
- **A dissimilar split is not always a harder one.** `split_report` measures
  how different the two splits are, but only training a model shows whether
  the split is harder (see the [examples](examples.md)).

## Split reports

`split_report` measures how different the training and test splits are. It
reports the centroid distance, coverage, cluster leakage, MMD, energy distance,
mean Wasserstein distance and mean KS statistic, and the shift in the label
distribution if you pass `y`. `compare_splitters` runs several splitters and
tabulates the results.

```python
from splytters import split_report

report = split_report(X, train_idx, test_idx, y=labels)
```

## scikit-learn integration

- **`SplytterSplit`**: a `BaseCrossValidator` that can be passed as `cv=` to
  `cross_validate`, `GridSearchCV` and other scikit-learn functions.
- **`splytter_train_test_split`** and the family wrappers
  **`adversarial_/overlap_/balanced_train_test_split`** mirror
  `sklearn.model_selection.train_test_split`: for inputs `a, b` they return
  `[a_train, a_test, b_train, b_test]`.

## Framework interop

The following helpers import their dependencies lazily, so the core install does
not require pandas, torch or `datasets`:

- **`split_dataframe`** (pandas): splits a `DataFrame`, optionally embedding a
  subset of its columns
- **`to_torch_subsets`** (torch): wraps a split as `torch.utils.data.Subset`s
- **`split_dataset`** (Hugging Face `datasets`): splits a `Dataset`

Splitters also accept torch tensors directly.

## Sorters

The `splytters.sorters` subpackage ranks samples by interpretable difficulty and
quality metrics. The rankings can be used for sorted splits ("train on easy,
test on hard") or to inspect a dataset:

```python
from splytters.sorters import distance_to_mean

ranked = distance_to_mean(embeddings)  # most typical → most atypical
```

Each modality requires an optional extra and is imported lazily:
`embedding_sorters` (core), `text_sorters` (`pip install "splytters[text]"`),
`image_sorters` (`[image]`), `audio_sorters` (`[audio]`) and
`tabular_sorters` (`[tabular]`).

The [sorter gallery](sorter_gallery.md) applies every sorter to a small example.

## Optional extras

| Extra | Adds |
| --- | --- |
| `text`, `image`, `audio`, `tabular` | modality sorters |
| `embedders` | `TextEmbedder`, CLIP, OpenAI embedders |
| `ann` | approximate-nearest-neighbor backend for large-n splits |
| `viz` | matplotlib |
| `all` | everything above |

See the [API reference](api.md) for full signatures and parameters.
