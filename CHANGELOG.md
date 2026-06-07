# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 0.2.0

This release aligns the API with scikit-learn and adds a first-class
interoperability and reporting layer. **It contains breaking changes**, made
deliberately before the API has downstream users.

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

### Added
- **scikit-learn compatibility** (`splitters.sklearn_api`):
  `SplytterSplit` cross-validator (`split`/`get_n_splits`) usable as `cv=` in
  `cross_validate`/`GridSearchCV`, plus `splytter_train_test_split` and
  `adversarial_/overlap_/balanced_train_test_split` convenience functions.
- **Framework interop** (`splitters.interop`, lazily imported): `split_dataframe`
  (pandas), `to_torch_subsets` (PyTorch), `split_dataset` (HuggingFace
  `datasets`). Splitters also accept torch tensors directly (CPU/GPU).
- **Split-quality report** (`splitters.report`): `split_report` (centroid /
  cross / coverage / cluster-leakage / MMD / energy distance / mean Wasserstein
  / mean KS, plus optional label-distribution shift) and `compare_splitters`.
- **`wasserstein_adversarial_split`** — Wasserstein nearest-neighbor adversarial
  split (Søgaard et al., EACL 2021), ported from the `wasserstein-splitting`
  branch and adapted to real-valued embeddings.
- **Validation harnesses** (`experiments/run_experiment.py`,
  `experiments/validate.py`): the latter sweeps 3 datasets (synthetic, vision,
  real text) × 4 model families, reports *balanced* accuracy + label-coverage
  diagnostics (so covariate shift is distinguished from class-dropping), and
  shows `split_report`'s energy distance predicts realized difficulty
  (Spearman ρ ≈ 0.65). Confirms the thesis is real and model-agnostic.
- `[ann]` extra (`pynndescent`) for approximate-nearest-neighbor backends.

### Fixed
- `sorters/__init__.py` now imports each modality **lazily** (PEP 562
  `__getattr__`), so `import sorters` no longer requires every optional
  dependency and each extra (`[image]`, `[audio]`, …) is self-sufficient.
- `multi_column_sort` no longer produces `NaN` scores when a column contains
  missing values (NaN cells map to the neutral midpoint).
- `compute_split_similarity` raises a clear error on empty train/test sets
  instead of crashing.
- `metrics.diversity_text`: removed the dead `datatype` parameter, computes over
  unique pairs (halving work), and gained optional subsampling for large inputs.

### Fixed (test suite / latest dependencies)
The suite is now fully green on current library versions (307 passed, incl. slow
model-download tests). Failures fell into three buckets:
- **Library API drift:** librosa now returns tempo as an array (`tempo` extracts
  the scalar); transformers ≥5 returns a `BaseModelOutputWithPooling` from CLIP
  `get_*_features` (`embedders._features_to_numpy` handles tensor *and* object);
  NLTK 3.9 renamed `punkt`→`punkt_tab` (readability tests ensure the resource).
- **Return types:** audio/image sorters returned `np.float32`; they now return
  Python `float` (matching their `-> tuple[int, float]` hints).
- **Broken fixtures:** the audio generator peak-normalized every clip (so
  "quiet"/"loud" were identical) and the "detailed" frequency image scored below
  "medium"; both generators are fixed (and seeded) and the data regenerated.
- **Code bug:** `readability_score` swallowed its own invalid-metric `ValueError`
  in a broad `except`; the metric is now validated up front.
- Model-download tests (`test_embedders`) are marked `slow`.

### Infrastructure
- GitHub Actions CI (core-only matrix on Python 3.10–3.12 + ruff lint +
  full-deps canary), `conftest.py` (seeding + optional-dep skipping),
  `CITATION.cff`, `CONTRIBUTING.md`, registered `slow` pytest marker, and ruff
  configuration.

## [0.1.0]
- Initial release: adversarial / overlap / balanced splitters, per-modality
  sorters, embedders, demos, and the pytest suite.
