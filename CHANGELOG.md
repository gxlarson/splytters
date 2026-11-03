# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`central_split`** (overlap family), the mirror of
  `distance_adversarial_split`: the samples closest to the centroid go to test
  and the rest to train, giving a prototypical, optimistic test set (#69).
  Unlike swapping the outputs of `distance_adversarial_split`, `train_size`
  means the training set, rounding matches every other splitter, and integer
  `train_size` works. Honors `metric`; wrap it in `per_class_split` for
  class-typical test samples.

### Fixed
- `split_report` no longer does O(n²) work in the full dataset. `max_samples`
  now caps the geometric metrics too (`mean_cross_distance`, `coverage`), not
  just the distribution metrics, so reports on large datasets finish instead of
  allocating an n × n matrix (1.28 TB at n = 400k) (#65).
- `compute_split_similarity` gains `max_samples` and `random_state`. With
  `max_samples`, the nearest-train statistics use a random subsample of test
  points (still searched against the full train set) and the median distance
  behind `coverage` is estimated from a random subsample. The default (`None`)
  stays exact and returns the same values as before. The nearest-train search
  is now chunked and multithreaded, so it also uses bounded memory and runs
  about 10× faster.
- `compute_split_similarity` and `split_report` warn when `max_samples=None`
  would need more than about 1 GiB of distance matrices, and suggest passing
  `max_samples`.
- `distance_adversarial_split` now uses its `metric` argument (#66). It was
  accepted but ignored, so every value, including invalid names, produced the
  euclidean split. Any `scipy.spatial.distance.cdist` metric (or callable) now
  ranks samples by distance from the centroid, and an unknown metric raises
  `ValueError`. The default euclidean split is unchanged.
- `deduplicated_split` and `duplicate_spread_split` no longer merge most of the
  dataset into one "near-duplicate" group under the default threshold (#68).
  `deduplicated_split` returned splits like 209 / 3791 for `train_size=0.8`,
  and `duplicate_spread_split` degraded to a random split. See **Changed** for
  the new default.
- Both functions find near-duplicates without building the n × n distance
  matrix, so memory now scales with the number of near-duplicate pairs.
  Exact duplicates are still detected exactly at `similarity_threshold=0`.
- `deduplicated_split` warns when indivisible near-duplicate groups push the
  split more than 5% of the samples away from `train_size`, reporting the group
  count and largest group. `duplicate_spread_split` warns when it finds no
  near-duplicates, or when one group holds more than half the samples, since
  either way its result is a random split.
- `cluster_split(method="dbscan")` no longer silently ignores `train_size`
  (#67). DBSCAN noise points are no longer treated as one indivisible cluster:
  `"size"` / `"centroid"` place them individually to fill train up to the
  target, and `"closest"` / `"subset_sum"` start them in train, where the
  individual fill / completion can use them. Previously a few near-duplicates
  on typical embeddings produced splits like 18 train / 15,982 test.
- `cluster_split` warns when train or test ends up smaller than half its
  requested size, reporting the cluster count and largest cluster and
  suggesting a fix (`eps` / `min_samples`, `n_clusters`, or
  `fill_individual=True`). This also covers imbalanced KMeans clusters.
- `cluster_split`, `cluster_kfold`, and `minority_split` warn when DBSCAN
  labels more than half the samples as noise.
- `cluster_kfold(method="dbscan")` no longer puts all DBSCAN noise points in
  one fold. They are placed one at a time after the whole clusters, wherever
  they best balance the folds' labels and sizes.
- `per_class_split` with `on_error="fallback"` no longer re-raises when an
  absolute-count `train_size` is greater than or equal to a class's size; the
  fallback clamps the count to the class's `[1, size - 1]` range so the class
  stays on both sides.
- `split_dataframe`, `to_torch_subsets`, and `split_dataset` (`splytters.interop`)
  only forward `random_state` to splitters that accept it, so a custom splitter
  without a `random_state` parameter no longer raises `TypeError`. They also
  accept splitters that return plain lists of indices.
- `nearest_neighbor_split` no longer records a point as its own nearest
  neighbor when there are exact duplicates, which wasted test capacity and
  raised spurious "could only place N" warnings.
- `column_rank` (tabular) now keeps NaN rows at the end for both sort
  directions. The previous `na_option='bottom'` gave NaN the maximum rank, so
  with `low_first=False` NaN rows sorted to the *front*; a direction-aware
  sentinel (matching `column_zscore`) now sorts them last regardless.
- `column_zscore` and `column_absolute_zscore` (tabular) return `0.0` for
  single-row input. pandas' ddof=1 `std()` on one row is `NaN`, which slipped
  past the `std == 0` guard and yielded `NaN` scores; the guard now also covers
  `NaN`.
- `outlier_score(method="zscore")` (tabular) maps the all-NaN z-scores of a
  constant column to `0.0` before averaging, so an all-constant frame scores a
  uniform `0.0` instead of returning all-`NaN` and emitting a `RuntimeWarning`.
- Lazy modality-sorter imports now raise an actionable error when the optional
  dependency is missing, naming the pip extra to install
  (e.g. `pip install splytters[image]`) instead of a bare
  `ModuleNotFoundError: No module named 'PIL'`.
- `OpenAIEmbedder.embed` sorts the API response by each item's `index` before
  stacking, so the returned rows stay aligned with the input order even if the
  API returns them out of order.

### Changed
- The default `similarity_threshold` of `deduplicated_split` and
  `duplicate_spread_split` is now 0.1 × the median distance from each sample to
  its 10th nearest neighbor, replacing the 1st and 10th percentiles of all
  pairwise distances. The old defaults linked each sample to a fixed fraction
  of all others, so their groups grew with the dataset. Splits made with the
  default threshold will differ; an explicit `similarity_threshold` behaves as
  before.
- For `method="dbscan"` in `cluster_split`, `cluster_kfold`, and
  `minority_split`, omitting `eps` now uses a default scaled to the data (the
  90th percentile of each sample's distance to its `(min_samples - 1)`-th
  nearest neighbor, so about 90% of samples are core points) instead of
  sklearn's absolute `eps=0.5`, which labels nearly every sample noise on
  unnormalized embeddings. Splits made without an explicit `eps` will differ.
- Contributor git hooks now run through pre-commit (`.pre-commit-config.yaml`):
  gitleaks secret scanning and ruff lint on every commit. This replaces the
  regex-based `scripts/hooks/pre-push` hook; run
  `git config --unset core.hooksPath` and then `pre-commit install`. CI gains a
  `secrets` job that runs gitleaks over the full history.
- `group_split` (and `deduplicated_split`, which shares the routine) can now
  put a group larger than the target train size in train when that lands
  closer to `train_size`. Previously such a group was always sent to test,
  badly undershooting `train_size` (e.g. 9% train for a requested 70%). The
  assignment runs both the old first-fit pass and a best-fit pass and keeps
  whichever is closer to the target, so it is never further off than before.
  Groups stay whole and both sides stay non-empty.

### Documentation
- `stratified_random_split` documents that every class in `y` needs at least 2
  members or scikit-learn raises `ValueError`.
- `histogram_matched_split` documents that an all-constant feature set makes the
  matching objective vacuous, leaving the result equivalent to a random split.

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
