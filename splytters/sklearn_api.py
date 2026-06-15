"""
scikit-learn compatibility layer for splytters.

Provides:

* :class:`SplytterSplit` — a single-split cross-validator wrapping any splytter
  splitting function, usable as ``cv=`` in :func:`sklearn.model_selection.cross_validate`
  and :class:`~sklearn.model_selection.GridSearchCV`.
* ``*_train_test_split`` convenience functions mirroring
  :func:`sklearn.model_selection.train_test_split` for drop-in swaps.

The functional splitters in :mod:`splitters` remain the source of truth; these
wrappers are purely additive.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from typing import Any

import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import _safe_indexing

from splytters._types import Splitter
from splytters.adversarial import cluster_split
from splytters.balanced import distribution_matched_split
from splytters.overlap import cluster_leak_split


def _derive_seed(random_state: int | None, i: int) -> int | None:
    """Derive a distinct, reproducible seed for the i-th repeat."""
    if random_state is None:
        return None
    return int(random_state) + i


class SplytterSplit(BaseCrossValidator):
    """A scikit-learn cross-validator backed by a splytter splitting function.

    Produces ``n_splits`` train/test partitions (one by default, like
    :class:`~sklearn.model_selection.PredefinedSplit`). Each partition is the
    output of ``splitter(embeddings, train_size=..., random_state=...)``.

    Args:
        splitter: any splytter splitter taking
            ``(embeddings, train_size=, random_state=)`` and returning
            ``(train_idx, test_idx)`` integer ndarrays. Defaults to
            :func:`~splytters.adversarial.cluster_split`.
        embeddings: array-like of shape (n_samples, n_features), optional.
            Embeddings to split on. If ``None``, the ``X`` passed to
            :meth:`split` is used (i.e. the estimator is assumed to consume the
            embeddings).
        train_size: fraction in (0, 1) or absolute count for the training set
            (default 0.7).
        random_state: base seed (default 42); the i-th repeat uses
            ``random_state + i``.
        n_splits: number of (repeated) partitions to yield (default 1).
        **splitter_kwargs: extra keyword arguments forwarded to ``splitter``.

    Examples:
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.model_selection import cross_validate
        >>> cv = SplytterSplit(embeddings=X)            # doctest: +SKIP
        >>> cross_validate(LogisticRegression(), X, y, cv=cv)   # doctest: +SKIP
    """

    def __init__(
        self,
        splitter: Splitter = cluster_split,
        *,
        embeddings: Any = None,
        train_size: float | int = 0.7,
        random_state: int | None = 42,
        n_splits: int = 1,
        **splitter_kwargs: Any,
    ) -> None:
        self.splitter = splitter
        self.embeddings = embeddings
        self.train_size = train_size
        self.random_state = random_state
        self.n_splits = n_splits
        self.splitter_kwargs = splitter_kwargs

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        emb = self.embeddings if self.embeddings is not None else X
        for i in range(self.n_splits):
            seed = _derive_seed(self.random_state, i)
            train_idx, test_idx = self.splitter(
                emb,
                train_size=self.train_size,
                random_state=seed,
                **self.splitter_kwargs,
            )
            yield np.asarray(train_idx, dtype=np.intp), np.asarray(
                test_idx, dtype=np.intp
            )

    def _iter_test_indices(self, X=None, y=None, groups=None):
        # Splytter splits are always a full partition, so test indices fully
        # determine the fold; this keeps the base-class helpers consistent.
        for _, test_idx in self.split(X, y, groups):
            yield test_idx


def splytter_train_test_split(
    *arrays: Any,
    splitter: Splitter = cluster_split,
    embeddings: Any = None,
    train_size: float | int = 0.7,
    random_state: int | None = 42,
    **splitter_kwargs: Any,
) -> list[Any] | tuple[np.ndarray, np.ndarray]:
    """Split arrays into train/test subsets using a splytter splitter.

    Mirrors :func:`sklearn.model_selection.train_test_split`: for inputs
    ``a, b`` returns ``[a_train, a_test, b_train, b_test]``.

    Args:
        *arrays: array-likes (numpy, list, pandas, etc.) to split, all of the
            same length (n_samples).
        splitter: splytter splitting function. Defaults to
            :func:`~splytters.adversarial.cluster_split`.
        embeddings: array-like to compute the split on, optional. Defaults to
            the first array.
        train_size: fraction in (0, 1) or absolute count for the training set.
        random_state: seed forwarded to ``splitter``.
        **splitter_kwargs: extra keyword arguments forwarded to ``splitter``.

    Returns:
        ``len(arrays) * 2`` outputs (train/test pairs in order). If no arrays
        are passed, returns ``(train_idx, test_idx)``.
    """
    if embeddings is None:
        if not arrays:
            raise ValueError("Pass at least one array or `embeddings=`.")
        embeddings = arrays[0]

    train_idx, test_idx = splitter(
        embeddings, train_size=train_size, random_state=random_state, **splitter_kwargs
    )

    if not arrays:
        return train_idx, test_idx

    out: list[Any] = []
    for a in arrays:
        out.append(_safe_indexing(a, train_idx))
        out.append(_safe_indexing(a, test_idx))
    return out


# Family-specific convenience wrappers (drop-in for train_test_split).
adversarial_train_test_split = partial(
    splytter_train_test_split, splitter=cluster_split
)
overlap_train_test_split = partial(
    splytter_train_test_split, splitter=cluster_leak_split
)
balanced_train_test_split = partial(
    splytter_train_test_split, splitter=distribution_matched_split
)
