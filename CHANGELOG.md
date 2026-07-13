# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Faithfulness fixes to the Züfle et al. (2023) cluster-split strategies and the
Godbole & Jia (2023) likelihood split after a re-audit against the source
papers and their reference code. Two changes affect results for existing
callers; see **Changed**.

### Changed
- `likelihood_split`: the default length metric for `length_buckets` (when
  bucketing from `texts`) is now whitespace token count, approximating the
  paper's NLTK `word_tokenize` count, instead of character count. Bucketed
  splits may differ from previous releases; pass
  `lengths=[len(t) for t in texts]` to restore the old character-based
  bucketing exactly.
- `cluster_split` with `strategy="subset_sum"` or `strategy="closest"` returns
  different (more paper-faithful) splits than previous releases:
  `"subset_sum"` now solves the multidimensional subset-sum exactly and always
  completes to the exact class-balanced target; `"closest"` now seeds the
  cluster farthest from the mean of centroids and grows by single-linkage,
  matching the reference code. The `"size"` and `"centroid"` strategies are
  unchanged.

### Fixed
- `cluster_split(cluster_range=...)` with `strategy="closest"` and
  `fill_individual=True` no longer degenerates to the smallest `k`: the
  k-search now scores the pre-fill assignment on the paper's criterion (fewest
  individually-added examples) instead of the topped-up test set.

### Added
- `cluster_split(fill_anchor=...)`: the paper's prose and its released code
  disagree on what the CLOSEST-SPLIT individual fill measures proximity to, so
  both are offered — `"cluster_centroids"` (default, the paper's wording:
  nearest selected test-cluster centroid) and `"test_mean"` (the released
  reference code: mean of all current test embeddings).

## [0.2.1] — 2026-07-06

Bug-fix roll-up from a full-package code review. Most changes are fixes to
existing functions; see **Changed** for the one behavior change that can affect
existing callers (`perplexity_score`).

### Fixed
- `ngram_jaccard_similarity` no longer raises `ZeroDivisionError` on texts
  shorter than the n-gram order (e.g. two-token strings), and now applies its
  `tokenizer` argument instead of ignoring it. This fixes a crash reachable from
  `split_report(texts=...)` on short texts.
- `split_report` clamps `n_clusters` to the sample count, so it no longer
  crashes on datasets smaller than the default 10 clusters.
- `nearest_neighbor_split` now honors its documented contract — every test
  point's nearest neighbor is guaranteed to be in train — by pinning neighbors
  into train, and warns when the requested test size cannot be met.
- `duplicate_spread_split` apportions singleton (non-duplicate) points across
  both sides instead of routing them all to train, so the realized split tracks
  `train_size`.
- Descending tabular sorts (`column_zscore` / `column_absolute_zscore` with
  `low_first=False`) keep NaN rows at the end instead of sorting them first.
- `outlier_score` (embedding and tabular) exposes `random_state` as a named
  parameter, so passing it no longer raises `TypeError: got multiple values for
  keyword 'random_state'`.
- `min_cut_split` warns when spectral eigendecomposition fails and it falls back
  to a random (non-adversarial) split, instead of doing so silently.
- `max_coverage_split` uses incremental coverage bookkeeping (O(n) swaps),
  replacing the previous O(n^4) worst case.
- `normalized_cut_split` orients the Fiedler vector's sign deterministically, so
  the split is reproducible across LAPACK builds/platforms.
- Image sorters close image file handles after loading, avoiding descriptor
  exhaustion on large image lists (notably on Windows).

### Changed
- **`perplexity_score` now raises `ValueError` when only one of `model` /
  `tokenizer` is supplied**, instead of silently pairing your object with a
  mismatched GPT-2 default. Pass both or neither.
- `splytter_train_test_split` and `SplytterSplit` no longer force `random_state`
  onto custom splitters that don't accept it (the callable is introspected
  first).
- Tabular row sorters (`missing_value_ratio`, `row_sparsity`,
  `row_distance_to_mean`) are vectorized — roughly 100x faster on wide frames.
- Build requires `setuptools>=77` (PEP 639 license string); the `audio` extra
  now declares `soundfile` explicitly.

## [0.2.0] — 2026-07-01

### Added
- **Grouped splitters** (new `grouped` family): `group_split` keeps every sample
  sharing a group id (user / document / source) on one side — the embedding
  analogue of scikit-learn's `GroupShuffleSplit` — and `deduplicated_split`
  keeps each discovered near-duplicate cluster on one side (the inverse of
  `duplicate_spread_split`).
- **`maximin_split`** — farthest-point (k-center) test selection, for a diverse,
  space-covering held-out set that never under-represents sparse regions.
- **New `split_report` similarity metrics**: `c2st_auc` (classifier two-sample
  test — how distinguishable train and test are), `frechet_distance` (FID-style),
  `sliced_wasserstein`, and `manifold_precision` / `manifold_recall` (k-NN
  support coverage).
- **New sorters**: `mahalanobis_distance_to_mean` and `knn_label_disagreement`
  (embedding), `gzip_complexity` (text), and `sharpness` (image).
- **Seed-stability notes** in every splitter docstring, classifying how much the
  held-out set changes across random seeds (deterministic / structure-stable /
  varies-like-random).

### Fixed
- `min_cut_split` (spectral) now orients the Fiedler vector's sign
  deterministically (like scikit-learn's `svd_flip`), so the split is
  reproducible — previously the arbitrary `eigsh` sign could flip the held-out
  set to a completely disjoint one between seeds.
- `per_class_split` now forwards `random_state` to the wrapped splitter, so its
  seed actually drives the per-class splits (previously it only seeded the rare
  fallback path, leaving the wrapped splitter pinned to its default seed).

### Changed
- Documentation is now hosted on Read the Docs
  ([splytters.readthedocs.io](https://splytters.readthedocs.io)); docstring URLs
  auto-link, and the README gained an Installation section and docs links.
- CI: added a wheel-install smoke test (with a pre-publish gate) and
  README-example API-drift tests.

## [0.1.1] — 2026-06-22

Documentation and packaging refresh: PyPI / docs / Python badges, an Installation
section and `pip install` one-liner, corrected License metadata, and Read the
Docs hosting.

## [0.1.0] — 2026-06-22

First PyPI release. Aligns the API with scikit-learn and adds a first-class
interoperability and reporting layer. **It contains breaking changes** vs the
pre-release, made deliberately before the API had downstream users.

### Changed (breaking)
- Renamed the split parameter `train_ratio` → **`train_size`** across every
  splitter, matching `sklearn.model_selection.train_test_split`. `train_size`
  now also accepts an **absolute integer count**, not just a fraction.
- Splitters now **return integer `numpy.ndarray` index pairs** instead of
  Python lists.
- Input validation is centralized in `validate_split_inputs`, which now runs
  `sklearn.utils.check_array` — **NaN/inf, ragged, and non-2-D inputs raise a
  clear `ValueError`** instead of silently propagating.
- `random_state` handling uses `sklearn.utils.check_random_state` (accepts
  `int`, `RandomState`, or `None`); `distance_adversarial_split` and
  `density_adversarial_split` gained a `random_state` parameter for API
  symmetry.
- `stratified_random_split`'s second argument was renamed `labels` → `y` to
  match scikit-learn's `split(X, y)` convention.
- **Single import namespace.** Everything now lives under the `splytters`
  package: `from splytters import cluster_split, SplytterSplit, split_report`,
  `from splytters.sorters import …`, `from splytters.embedders import …`. The
  former top-level `splitters` / `sorters` / `embedders` import names are gone
  (they squatted generic namespace and the dist name `splytters` didn't match
  the import name `splitters`).

### Added
- **scikit-learn compatibility** (`splytters.sklearn_api`):
  `SplytterSplit` cross-validator (`split`/`get_n_splits`) usable as `cv=` in
  `cross_validate`/`GridSearchCV`, plus `splytter_train_test_split` and
  `adversarial_/overlap_/balanced_train_test_split` convenience functions.
- **Framework interop** (`splytters.interop`, lazily imported): `split_dataframe`
  (pandas), `to_torch_subsets` (PyTorch), `split_dataset` (HuggingFace
  `datasets`). Splitters also accept torch tensors directly (CPU/GPU).
- **Split-quality report** (`splytters.report`): `split_report` (centroid /
  cross / coverage / cluster-leakage / MMD / energy distance / mean Wasserstein
  / mean KS, plus optional label-distribution shift) and `compare_splitters`.
- **`wasserstein_adversarial_split`** — Wasserstein nearest-neighbor adversarial
  split (Søgaard et al., EACL 2021), ported from the `wasserstein-splitting`
  branch and adapted to real-valued embeddings.
- **Supervised splitters** including `class_boundary_split`, `minority_split`,
  `minority_grow_split`, and `decision_boundary_split` (a learned-boundary margin
  split with a linear or RBF surrogate).
- `[ann]` extra (`pynndescent`) for approximate-nearest-neighbor backends.

### Fixed
- `splytters/sorters/__init__.py` now imports each modality **lazily** (PEP 562
  `__getattr__`), so `import splytters.sorters` no longer requires every optional
  dependency and each extra (`[image]`, `[audio]`, …) is self-sufficient.
- `multi_column_sort` no longer produces `NaN` scores when a column contains
  missing values (NaN cells map to the neutral midpoint).
- `compute_split_similarity` raises a clear error on empty train/test sets
  instead of crashing.
- Several correctness fixes in the bin/cluster splitters (empty-test-set and
  train-fraction bugs), `resolve_n_train` clamping, `histogram_matched_split`
  NaN on constant dimensions, `dynamic_range` (audio), and `dominant_color`
  (image) determinism; plus defensive guards for empty CV folds and array-length
  mismatches.
- `splytters.metrics.diversity_text`: computes over unique pairs (halving work)
  and gained optional subsampling for large inputs.

### Infrastructure
- GitHub Actions CI (core matrix on Python 3.10–3.14 + ruff lint + full-deps
  canary), Codecov coverage (~94%), Trusted-Publishing release workflow,
  `conftest.py` (seeding + optional-dep skipping), `CITATION.cff`,
  `CONTRIBUTING.md`, registered `slow` pytest marker, and ruff configuration.

## [0.0.x] — pre-release (2022–2026)

The original repository before the scikit-learn API alignment and PyPI
packaging: adversarial / overlap / balanced splitters, per-modality sorters,
embedders, demos, and the pytest suite.

[0.2.1]: https://github.com/gxlarson/splytters/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/gxlarson/splytters/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/gxlarson/splytters/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/gxlarson/splytters/releases/tag/v0.1.0
