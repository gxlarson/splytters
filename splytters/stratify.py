"""Apply any embedding splitter or sorter independently within each class.

Global splitters/sorters measure each sample against the whole dataset, which on
a labeled task can wipe whole classes out of train (a splitter) or rank by
typical-overall rather than typical-for-its-class (a sorter). Both are fixed by
operating per class: :func:`per_class_split` returns a ``(train_idx, test_idx)``
partition, :func:`per_class_sort` a within-class ``[(idx, score), ...]`` ranking
for :func:`splytters.sorted_stratified_split`. Every class with >= 1 sample stays
in train (coverage 1.0), so only pure within-class structure is left.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from splytters._types import Splitter
from splytters.utils import as_index_array, random_split, to_numpy


def _class_groups(y: np.ndarray) -> Iterator[tuple[Any, np.ndarray]]:
    """Yield ``(label, global_indices)`` for each unique label in ``y``."""
    for c in np.unique(y):
        yield c, np.flatnonzero(y == c)


def _accepts_random_state(fn: Any) -> bool:
    """Whether ``fn`` takes a ``random_state`` keyword (directly or via **kwargs)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return True  # can't introspect (e.g. a C callable) — assume it does
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return True
    return "random_state" in params


def _take(data: Any, idx: np.ndarray) -> Any:
    """Subset ``data`` by integer indices, for arrays/tensors or plain sequences."""
    if hasattr(data, "ndim"):  # ndarray / torch tensor — fancy-index directly
        return data[idx]
    return [data[i] for i in idx]


def per_class_split(
    split_fn: Splitter,
    embeddings: ArrayLike,
    y: ArrayLike,
    train_size: float | int = 0.7,
    *,
    on_error: str = "fallback",
    random_state: int = 42,
    **split_kwargs: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply ``split_fn`` independently within each class and concatenate.

    Pair any embedding splitter (e.g. ``distance_adversarial_split``,
    ``cluster_leak_split``) with class labels to get a stratified version of it:
    the splitter's objective is pursued separately inside each class's own
    sub-cloud, so no class is wiped out of the training set.

    Args:
        split_fn: a base splitter with the standard contract
            ``split_fn(embeddings, train_size, **kwargs) -> (train_idx, test_idx)``,
            returning integer indices into the embeddings it was given.
        embeddings: array-like of shape (n_samples, n_features), aligned to ``y``.
        y: class labels aligned to ``embeddings`` (length n_samples). The split
            is performed independently within each unique label.
        train_size: fraction in (0, 1) or absolute count, applied *per class*
            (an absolute count means that many train samples in every class).
        on_error: what to do when ``split_fn`` raises for a class (typically a
            class too small for the splitter's clustering, e.g. fewer samples
            than ``n_clusters``). ``"fallback"`` (default) falls back to a random
            split for that class so coverage is preserved; ``"raise"`` propagates
            the original error.
        random_state: seed forwarded to ``split_fn`` on every per-class call (so
            it actually drives the wrapped splitter), and used for the
            ``"fallback"`` random split. Deterministic splitters ignore it; a
            custom ``split_fn`` that doesn't accept ``random_state`` is called
            without it.
        **split_kwargs: forwarded to ``split_fn`` on every per-class call.

    Returns:
        ``(train_indices, test_indices)`` integer ndarrays into the original
        sample order, sorted ascending.

    Raises:
        ValueError: if ``embeddings`` and ``y`` differ in length, or ``on_error``
            is not ``"fallback"`` or ``"raise"``.
    """
    if on_error not in ("fallback", "raise"):
        raise ValueError(f"on_error must be 'fallback' or 'raise', got {on_error!r}")

    X = to_numpy(embeddings)
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError(f"embeddings has {len(X)} rows but y has {len(y)} labels")

    # Forward the seed to split_fn so per_class_split's random_state actually
    # drives the wrapped splitter (deterministic splitters ignore it). Custom
    # split_fns that don't accept random_state are called without it.
    call_kwargs = dict(split_kwargs)
    if _accepts_random_state(split_fn):
        call_kwargs.setdefault("random_state", random_state)

    train: list[int] = []
    test: list[int] = []
    for _c, idx in _class_groups(y):
        # A singleton class can't appear on both sides; keep it seen in train.
        if len(idx) < 2:
            train.extend(idx.tolist())
            continue

        class_emb = X[idx]
        try:
            tr_local, te_local = split_fn(class_emb, train_size, **call_kwargs)
        except Exception:
            if on_error == "raise":
                raise
            tr_local, te_local = random_split(class_emb, train_size, random_state)

        train.extend(idx[np.asarray(tr_local, dtype=np.intp)].tolist())
        test.extend(idx[np.asarray(te_local, dtype=np.intp)].tolist())

    return as_index_array(sorted(train)), as_index_array(sorted(test))


def per_class_sort(
    sort_fn: Any,
    data: Sequence[Any] | ArrayLike,
    y: ArrayLike,
    *,
    on_error: str = "fallback",
    **sort_kwargs: object,
) -> list[tuple[int, float]]:
    """Apply a sorter independently within each class and combine the rankings.

    The embedding sorters score each sample against the *whole* dataset (e.g.
    ``distance_to_mean`` uses the global centroid). Wrapping them here recomputes
    the ranking inside each class, so "easy" means typical *for that intent*. The
    result has the same ``[(index, score), ...]`` shape as a plain sorter and is
    a drop-in ``order`` for :func:`splytters.sorted_stratified_split`, which only
    relies on the relative order *within* each class.

    Text sorters whose scores are intrinsic per-sample properties (length,
    readability, rarity) are unaffected by this -- their global ranking already
    restricts to the same within-class order -- so this only changes behavior for
    dataset-relative sorters (the embedding ones).

    Args:
        sort_fn: a sorter with the contract ``sort_fn(data, **kwargs) ->
            [(index, score), ...]``, where ``index`` is into the data it was
            given. Works for embedding sorters (``data`` an array) and text
            sorters (``data`` a list of strings).
        data: the samples to rank, aligned to ``y`` (an array of embeddings, a
            list of texts, etc.).
        y: class labels aligned to ``data`` (length n_samples).
        on_error: what to do when ``sort_fn`` raises for a class. ``"fallback"``
            (default) keeps that class's samples in their original order;
            ``"raise"`` propagates the error.
        **sort_kwargs: forwarded to ``sort_fn`` on every per-class call.

    Returns:
        A flat ``[(index, score), ...]`` ranking over the original sample order:
        each class's members in that class's sorted order, classes concatenated
        in label order. It is a permutation of ``range(n_samples)`` by index, so
        it is a valid ``order`` for ``sorted_stratified_split``.

    Raises:
        ValueError: if ``data`` and ``y`` differ in length, or ``on_error`` is
            not ``"fallback"`` or ``"raise"``.
    """
    if on_error not in ("fallback", "raise"):
        raise ValueError(f"on_error must be 'fallback' or 'raise', got {on_error!r}")

    data = to_numpy(data)
    y = np.asarray(y)
    if len(data) != len(y):
        raise ValueError(f"data has {len(data)} items but y has {len(y)} labels")

    order: list[tuple[int, float]] = []
    for _c, idx in _class_groups(y):
        try:
            ranked = sort_fn(_take(data, idx), **sort_kwargs)
        except Exception:
            if on_error == "raise":
                raise
            ranked = [(i, float("nan")) for i in range(len(idx))]

        # Remap local indices to global, de-duplicating defensively.
        seen: set[int] = set()
        for local_idx, score in ranked:
            li = int(local_idx)
            if li in seen:
                continue
            seen.add(li)
            order.append((int(idx[li]), float(score)))
        # Any class members the sorter omitted go last, in original order, so the
        # combined ranking stays a full permutation (sorted_stratified_split needs it).
        for li in range(len(idx)):
            if li not in seen:
                order.append((int(idx[li]), float("nan")))

    return order
