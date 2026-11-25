# TODO

## sklearn interop

Make the splytters API coalesce with scikit-learn so splitters can drop into existing workflows.

- [x] **CV splitter protocol wrappers** — `SplytterSplit` (`splytters/sklearn_api.py`) implements `split`/`get_n_splits` and works as `cv=` in `cross_validate`/`GridSearchCV`.
- [x] **`train_test_split`-style convenience** — `splytter_train_test_split` + `adversarial_/overlap_/balanced_train_test_split` slice every passed array via `_safe_indexing`.
- [x] **Align conventions with sklearn** — `train_ratio` → `train_size` (fraction *or* int count); splitters return ndarray indices; `check_array`/`check_random_state` validation; `stratified_random_split(..., y=...)`.

## Other library interop

- [x] **PyTorch** — splitters accept torch tensors (CPU/GPU, via `to_numpy`); `to_torch_subsets` returns `Subset` pairs.
- [x] **Pandas** — `split_dataframe(df, embeddings, ...)` returns two DataFrames via `.iloc`.
- [x] **HuggingFace datasets** — `split_dataset(ds, embeddings, ...)` returns a `DatasetDict`.
- [x] **Sparse embeddings** — `scipy.sparse` input is now rejected with a clear error via `check_array` (dense ANN support is future work).

Not planned: **Polars** input. Splitters take embedding arrays, so a polars user can split with `df[train_idx]`; full support would also mean a second code path in every tabular sorter. Revisit if users ask for it.

## Scalability

- [~] **Approximate / chunked nearest neighbors** — `embedding_sorters.distance_to_nearest_neighbor`, `knn_label_disagreement`, and both `density_*_split` functions now use exact `NearestNeighbors` queries with O(n·k) result storage; `[ann]` extra (`pynndescent`) added. Remaining O(n²): `local_density`, `min_cut_split`, `normalized_cut_split`, `neighbor_coverage_split`, `max_coverage_split`, `maximin_split`, and `utils.compute_pairwise_distances` (each still TODO-flagged in-code). `deduplicated_split` / `duplicate_spread_split` use O(n · k) neighbor queries plus memory proportional to the near-duplicate pairs. `compute_split_similarity` is O(max_samples · n) when `max_samples` is set (as `split_report` does).

## Infra

- [x] **CI** — GitHub Actions: core-only matrix (Python 3.10–3.14, plus Windows on 3.12), ruff lint, mypy, a full-deps canary, a wheel-install smoke test and gitleaks; CodeQL and dependency review run as separate workflows. `conftest.py` seeds RNG and skips heavy-modality test modules when their optional dep is absent.
- [x] **Split-quality report** — `split_report` / `compare_splitters` (`splytters/report.py`): geometric + cluster-leakage + MMD/energy/Wasserstein/KS + label-shift metrics.
- [x] **Heavy-modality test drift** — fixed for the latest librosa/transformers/NLTK; also fixed two broken data generators (loudness/frequency fixtures) and a swallowed `ValueError` in `readability_score`. `readability_score` now scores only too-short texts as `inf` and lets other errors, such as missing NLTK data, propagate (#89).
- [x] **Docs site** — Material for MkDocs (`mkdocs.yml`, `docs/`) with a docstring-driven API reference via `mkdocstrings`, end-to-end examples (`docs/examples/`) and a generated sorter gallery. Hosted on Read the Docs (https://splytters.readthedocs.io, configured by `.readthedocs.yaml`); `.github/workflows/docs.yml` only checks that `mkdocs build --strict` succeeds. Build locally with `pip install -e ".[docs]" && mkdocs serve`.
- [ ] **Sorter fixes** — items 2–4 of #84: `vocabulary_rarity` barely separates rare from common words, `mahalanobis_distance_to_mean` returns equal distances when there are fewer samples than dimensions, and `categorical_rarity` reports frequency rather than rarity. Items 2 and 4 change scores, so they need a changelog note.
- [ ] **Zenodo DOI** — archive a tagged release; fill `doi:` in `CITATION.cff` + README badge.

## Research

- [x] **Illustrative experiment harness** — `run_experiment.py` (in the companion [`splytters-paper`](../splytters-paper) repo): offline `digits` shows a large accuracy drop under an adversarial split vs random (overlap inflates, balanced matches); MMD tracks difficulty.
- [x] **Multi-dataset / multi-model evaluation** — `validate.py` (in [`splytters-paper`](../splytters-paper)): synthetic + vision + real text × 4 model families; adversarial splits harder everywhere (no class dropped); `split_report` energy predicts the drop (Spearman ρ≈0.65). (Experiments + paper write-up are kept outside this repo.)
- [x] **Reconcile `wasserstein-splitting` branch** — ported as `wasserstein_adversarial_split` (Søgaard et al., EACL 2021).

## Packaging

- [x] Add `pyproject.toml` with core deps and optional extras (`[text]`, `[audio]`, `[image]`, `[tabular]`, `[embedders]`, `[viz]`, `[ann]`, `[demo]`, `[all]`, `[dev]`), enabling `pip install -e .`.
- [x] **Lazy imports in `splytters/sorters/__init__.py`** — per-modality PEP 562 `__getattr__`; each extra is now self-sufficient.
- [ ] **Bump torch for transformers 5.x** — `transformers>=5` requires `torch>=2.4`. The PyTorch-backed features (`perplexity_score` sorter, CLIP embedders) and their tests are disabled on environments with older torch (e.g. a dev box on torch 2.1). Action: upgrade local/dev torch to `>=2.4`, and consider adding `torch>=2.4` lower bounds to the `[text]`/`[embedders]` extras so installs stay consistent with `transformers>=5`.
