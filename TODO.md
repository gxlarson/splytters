# TODO

## sklearn interop

Make the splytters API coalesce with scikit-learn so splitters can drop into existing workflows.

- [x] **CV splitter protocol wrappers** — `SplytterSplit` (`splitters/sklearn_api.py`) implements `split`/`get_n_splits` and works as `cv=` in `cross_validate`/`GridSearchCV`.
- [x] **`train_test_split`-style convenience** — `splytter_train_test_split` + `adversarial_/overlap_/balanced_train_test_split` slice every passed array via `_safe_indexing`.
- [x] **Align conventions with sklearn** — `train_ratio` → `train_size` (fraction *or* int count); splitters return ndarray indices; `check_array`/`check_random_state` validation; `stratified_random_split(..., y=...)`.

## Other library interop

- [x] **PyTorch** — splitters accept torch tensors (CPU/GPU, via `to_numpy`); `to_torch_subsets` returns `Subset` pairs.
- [x] **Pandas** — `split_dataframe(df, embeddings, ...)` returns two DataFrames via `.iloc`.
- [x] **HuggingFace datasets** — `split_dataset(ds, embeddings, ...)` returns a `DatasetDict`.
- [x] **Sparse embeddings** — `scipy.sparse` input is now rejected with a clear error via `check_array` (dense ANN support is future work).
- [ ] **Polars** — accept polars DataFrames/Series as input alongside pandas.

## Scalability

- [~] **Approximate / chunked nearest neighbors** — `embedding_sorters.distance_to_nearest_neighbor` now uses `NearestNeighbors` (O(n·k)); `[ann]` extra (`pynndescent`) added. Remaining O(n²): `local_density`, `density_*_split`, `min_cut_split`, `normalized_cut_split`, `neighbor_coverage_split`, `duplicate_spread_split`, `max_coverage_split`, `compute_split_similarity` (each still TODO-flagged in-code).

## Infra

- [x] **CI** — GitHub Actions: core-only matrix (3.10–3.12) + ruff lint + full-deps canary. `conftest.py` seeds RNG and skips heavy-modality test modules when their optional dep is absent.
- [x] **Split-quality report** — `split_report` / `compare_splitters` (`splitters/report.py`): geometric + cluster-leakage + MMD/energy/Wasserstein/KS + label-shift metrics.
- [x] **Heavy-modality test drift** — fixed for the latest librosa/transformers/NLTK; also fixed two broken data generators (loudness/frequency fixtures) and a swallowed `ValueError` in `readability_score`. Suite is green (307 passed incl. slow; 294 in the default `-m "not slow"` scope).
- [ ] **Docs site** — mkdocs-material/Sphinx API reference + "reproduce the paper" page.
- [ ] **Zenodo DOI** — archive a tagged release; fill `doi:` in `CITATION.cff` + README badge.

## Research / paper

- [x] **Illustrative experiment harness** — `experiments/run_experiment.py` (+ `make figures`): offline `digits` shows a **38-pt** accuracy drop under an adversarial split vs random (overlap inflates, balanced matches); MMD tracks difficulty.
- [x] **Demo-paper draft** — `paper/splytters.md`.
- [ ] **Multi-dataset / multi-model evaluation** (text + vision) for a fuller study.
- [x] **Reconcile `wasserstein-splitting` branch** — ported as `wasserstein_adversarial_split` (Søgaard et al., EACL 2021).

## Packaging

- [x] Add `pyproject.toml` with core deps and optional extras (`[text]`, `[audio]`, `[image]`, `[tabular]`, `[embedders]`, `[viz]`, `[ann]`, `[demo]`, `[all]`, `[dev]`), enabling `pip install -e .`.
- [x] **Lazy imports in `sorters/__init__.py`** — per-modality PEP 562 `__getattr__`; each extra is now self-sufficient.
