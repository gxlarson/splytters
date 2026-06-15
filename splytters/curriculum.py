"""
Curriculum / ordering-driven splitting.

Unlike the embedding-based splitters (adversarial / overlap / balanced), these
take an explicit sort order — typically a :mod:`splytters.sorters` ranking — and
partition each class along it. The classic use is a "train on easy, test on
hard" curriculum split: sort by a difficulty metric, then take the first
``train_size`` fraction of each class as train.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from splytters.utils import as_index_array, resolve_n_train


def _to_index_order(order: Sequence[Any]) -> np.ndarray:
    """Coerce ``order`` to a 1-D array of sample indices.

    Accepts either a flat sequence of indices, or a sorter result — a sequence
    of ``(index, score)`` pairs (e.g. the output of ``readability_score``).
    """
    seq = list(order)
    if not seq:
        return np.empty(0, dtype=np.intp)
    first = seq[0]
    if isinstance(first, (tuple, list, np.ndarray)) and not np.isscalar(first):
        idx = [int(item[0]) for item in seq]  # (index, score) pairs from a sorter
    else:
        idx = [int(i) for i in seq]
    return np.asarray(idx, dtype=np.intp)


def sorted_stratified_split(
    order: Sequence[Any],
    y: ArrayLike,
    train_size: float | int = 0.7,
    *,
    largest_first: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class curriculum split driven by a sort order.

    Within each class, samples are visited in the given ``order`` and the first
    ``train_size`` fraction (or absolute count) go to train, the rest to test.
    Pair it with a :mod:`splytters.sorters` ranking to split along an
    interpretable difficulty axis — e.g. "train on easy, test on hard".

    Args:
        order: the sort order over samples — either a flat sequence of sample
            indices, or a sorter result (sequence of ``(index, score)`` pairs,
            as returned by e.g. ``readability_score``). Sorters rank ascending,
            so the lowest-scoring samples are the train-preferred "first" ones.
        y: class labels aligned to the original samples (length n_samples). The
            split is performed independently within each class.
        train_size: fraction in (0, 1) or absolute count, applied *per class*.
        largest_first: if True, reverse ``order`` so the highest-scoring samples
            are train-preferred.

    Returns:
        ``(train_indices, test_indices)`` integer ndarrays, sorted ascending.

    Raises:
        ValueError: if ``order`` and ``y`` differ in length, ``order`` is not a
            permutation of ``range(len(y))``, or ``train_size`` is out of range.
    """
    ordered_idx = _to_index_order(order)
    y = np.asarray(y)

    if len(ordered_idx) != len(y):
        raise ValueError(f"order has {len(ordered_idx)} entries but y has {len(y)}")
    if len(ordered_idx) and (
        ordered_idx.min() < 0
        or ordered_idx.max() >= len(y)
        or len(np.unique(ordered_idx)) != len(ordered_idx)
    ):
        raise ValueError("order must be a permutation of range(len(y))")
    if isinstance(train_size, float) and not 0.0 < train_size < 1.0:
        raise ValueError(f"train_size fraction must be in (0, 1), got {train_size}")

    if largest_first:
        ordered_idx = ordered_idx[::-1]

    labels = y[ordered_idx]  # class labels in sorted order
    train: list[int] = []
    test: list[int] = []
    for c in np.unique(y):
        class_order = ordered_idx[labels == c]  # this class, in sorted order
        n_train = max(0, min(resolve_n_train(len(class_order), train_size), len(class_order)))
        train.extend(class_order[:n_train].tolist())
        test.extend(class_order[n_train:].tolist())

    return as_index_array(sorted(train)), as_index_array(sorted(test))
