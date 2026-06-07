# TODO

## sklearn interop

Make the splytters API coalesce with scikit-learn so splitters can drop into existing workflows.

- [ ] **CV splitter protocol wrappers** — class wrappers implementing `split(X, y=None, groups=None)` (yields `(train_idx, test_idx)` ndarrays) and `get_n_splits()`, so splitters work as `cv=` in `GridSearchCV` / `cross_validate`. Single-split CV objects are an established sklearn pattern (`PredefinedSplit`, `ShuffleSplit(n_splits=1)`). The functional core stays as-is; wrappers are additive.
- [ ] **`train_test_split`-style convenience function** — accept `*arrays` and return split data directly, e.g. `adversarial_train_test_split(texts, labels, embeddings=emb, train_size=0.7)`, mirroring `sklearn.model_selection.train_test_split` for drop-in swaps.
- [ ] **Align conventions with sklearn** — rename `train_ratio` → `train_size`; return ndarray indices instead of lists. Worth doing before the API has users.

## Other library interop

- [ ] **PyTorch** — accept torch tensors as embeddings input (GPU tensors currently fail in `np.asarray`; needs a `.cpu()` conversion path). Add a helper that returns `torch.utils.data.Subset` pairs from a Dataset + embeddings.
- [ ] **Pandas** — `split_dataframe(df, embeddings, ...)` returning two DataFrames via `.iloc`. (`sorters/tabular_sorters.py` already uses pandas; splitters have no DataFrame story yet.)
- [ ] **HuggingFace datasets** — helper returning `DatasetDict({"train": ds.select(train_idx), "test": ds.select(test_idx)})`; would also simplify `demo.py`.
- [ ] **Sparse embeddings** — `scipy.sparse` input (e.g. TF-IDF) is silently mangled by `np.asarray`; either support it or reject it with a clear error.
- [ ] **Polars** — accept polars DataFrames/Series as input alongside pandas.

## Scalability

- [ ] **Approximate nearest neighbors backend** — `compute_pairwise_distances` materializes a full O(n²) matrix (already flagged in its docstring). Offer faiss/annoy/pynndescent for large datasets.

## Infra

- [ ] **CI** — GitHub Actions workflow running the pytest suite (94 tests) on push/PR.
- [ ] **Split-quality report** — build on `compute_split_similarity` to expose a summary of how adversarial/overlapping/balanced a produced split actually is, to help users compare splitters.

## Packaging

- [ ] Add `pyproject.toml` with core deps (`numpy`, `scipy`, `scikit-learn`) and optional extras (`[text]`, `[audio]`, `[image]`, `[demo]`), enabling `pip install -e .`; simplify the README install section accordingly.
