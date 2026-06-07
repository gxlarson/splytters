"""
Framework interop helpers for splytters.

Thin adapters that take an embedding-based splitter and return native objects
for popular ecosystems. Heavy dependencies (pandas, torch, datasets) are
imported lazily inside each function, so importing this module never requires
them — only calling the relevant helper does.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from splitters.adversarial import cluster_split

Splitter = Callable[..., tuple[np.ndarray, np.ndarray]]


def split_dataframe(
    df: Any,
    embeddings: Any,
    splitter: Splitter = cluster_split,
    *,
    train_size: float | int = 0.7,
    random_state: int | None = 42,
    **splitter_kwargs: Any,
) -> tuple[Any, Any]:
    """Split a pandas DataFrame into (train_df, test_df) by position.

    Parameters
    ----------
    df : pandas.DataFrame of n_samples rows.
    embeddings : array-like of shape (n_samples, n_features).
    splitter : splytter splitting function.
    train_size, random_state, **splitter_kwargs : forwarded to ``splitter``.

    Returns
    -------
    (train_df, test_df) : selected via positional ``.iloc`` (original index
    labels are preserved).
    """
    if len(df) != len(embeddings):
        raise ValueError(
            f"df has {len(df)} rows but embeddings has {len(embeddings)}"
        )
    train_idx, test_idx = splitter(
        embeddings, train_size=train_size, random_state=random_state, **splitter_kwargs
    )
    return df.iloc[train_idx], df.iloc[test_idx]


def to_torch_subsets(
    dataset: Any,
    embeddings: Any,
    splitter: Splitter = cluster_split,
    *,
    train_size: float | int = 0.7,
    random_state: int | None = 42,
    **splitter_kwargs: Any,
) -> tuple[Any, Any]:
    """Split a torch ``Dataset`` into ``(train_subset, test_subset)``.

    Returns two :class:`torch.utils.data.Subset` views over ``dataset``.
    """
    from torch.utils.data import Subset

    if len(dataset) != len(embeddings):
        raise ValueError(
            f"dataset has {len(dataset)} items but embeddings has {len(embeddings)}"
        )
    train_idx, test_idx = splitter(
        embeddings, train_size=train_size, random_state=random_state, **splitter_kwargs
    )
    return (
        Subset(dataset, train_idx.tolist()),
        Subset(dataset, test_idx.tolist()),
    )


def split_dataset(
    ds: Any,
    embeddings: Any,
    splitter: Splitter = cluster_split,
    *,
    train_size: float | int = 0.7,
    random_state: int | None = 42,
    **splitter_kwargs: Any,
) -> Any:
    """Split a HuggingFace ``datasets.Dataset`` into a ``DatasetDict``.

    Returns ``DatasetDict({"train": ds.select(train_idx),
    "test": ds.select(test_idx)})``.
    """
    from datasets import DatasetDict

    if len(ds) != len(embeddings):
        raise ValueError(
            f"dataset has {len(ds)} rows but embeddings has {len(embeddings)}"
        )
    train_idx, test_idx = splitter(
        embeddings, train_size=train_size, random_state=random_state, **splitter_kwargs
    )
    return DatasetDict(
        {
            "train": ds.select(train_idx.tolist()),
            "test": ds.select(test_idx.tolist()),
        }
    )
