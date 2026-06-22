"""
Shared utilities for splitting algorithms.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.utils import check_array, check_random_state


def to_numpy(X: ArrayLike) -> Any:
    """Best-effort conversion of framework tensors to numpy.

    Handles PyTorch tensors (including those on GPU / requiring grad) by
    detaching, moving to CPU, and converting. Any other input is returned
    unchanged so the caller's downstream ``check_array``/``np.asarray`` can
    handle lists, numpy arrays, pandas DataFrames, etc.
    """
    # Duck-type torch.Tensor without importing torch.
    if hasattr(X, "detach") and hasattr(X, "cpu") and hasattr(X, "numpy"):
        try:
            return X.detach().cpu().numpy()
        except Exception:  # pragma: no cover - fall through to generic handling
            pass
    return X


def validate_split_inputs(
    embeddings: ArrayLike, train_size: float | int, min_samples: int = 2
) -> np.ndarray:
    """Validate and coerce inputs shared by every splitting function.

    Accepts any array-like (numpy, list, pandas, torch tensor), converts it to
    a finite 2-D float ``ndarray``, and validates ``train_size``.

    Args:
        embeddings: array-like of shape (n_samples, n_features).
        train_size: fraction in the open interval (0, 1), or an absolute count
            in ``[1, n_samples)``. Mirrors
            ``sklearn.model_selection.train_test_split``.
        min_samples: minimum number of samples required to form a split
            (default 2).

    Returns:
        The validated, finite, float embeddings as an ndarray of shape
        (n_samples, n_features).

    Raises:
        ValueError: if ``train_size`` is out of range, there are too few
            samples, or the embeddings contain NaN/inf or are not 2-D.
    """
    # Validate train_size first so its message takes priority (matches the
    # historical "between 0 and 1" contract for fractional sizes).
    is_int = isinstance(train_size, (int, np.integer)) and not isinstance(
        train_size, bool
    )
    if not is_int:
        if not (isinstance(train_size, (float, np.floating)) and 0 < train_size < 1):
            raise ValueError(
                "train_size as a fraction must be between 0 and 1 exclusive, "
                f"got {train_size!r}"
            )

    X = to_numpy(embeddings)
    n = len(X)
    if n < min_samples:
        raise ValueError(f"Need at least {min_samples} samples to split, got {n}")

    if is_int and not (1 <= train_size < n):
        raise ValueError(
            f"train_size as an absolute count must be in [1, {n}), got {train_size}"
        )

    # Coerce to a finite 2-D numeric array (rejects NaN/inf, 1-D, ragged, sparse).
    X = check_array(X, ensure_2d=True, allow_nd=False, dtype="numeric")
    return X


def resolve_n_train(n_samples: int, train_size: float | int) -> int:
    """Resolve ``train_size`` (fraction or absolute count) to an int count.

    A fractional ``train_size`` is clamped to ``[1, n_samples - 1]`` so the
    resolved count never collapses one side of the split to empty (e.g.
    ``n_samples=2, train_size=0.3`` would otherwise truncate to 0).
    """
    if isinstance(train_size, (int, np.integer)) and not isinstance(train_size, bool):
        return int(train_size)
    n_train = int(n_samples * train_size)
    return min(max(n_train, 1), n_samples - 1)


def apportion_train(bin_sizes: list[int], n_train_total: int) -> np.ndarray:
    """Split ``n_train_total`` train slots across bins via largest-remainder.

    Each bin gets ``floor(size * n_train_total / total)`` slots, then the
    leftover slots go to the bins with the largest fractional remainders. The
    per-bin counts sum exactly to ``n_train_total`` (clamped to ``[0, total]``),
    avoiding both the per-bin ``int()`` truncation bias that undershoots the
    requested train fraction and the all-to-train rounding that can empty the
    test set.

    Args:
        bin_sizes: number of samples in each (non-empty) bin.
        n_train_total: total samples to assign to train across all bins.

    Returns:
        Integer ndarray of per-bin train counts, aligned with ``bin_sizes``.
    """
    sizes = np.asarray(bin_sizes, dtype=np.intp)
    total = int(sizes.sum())
    if total == 0:
        return np.zeros(len(sizes), dtype=np.intp)
    n_train_total = int(min(max(n_train_total, 0), total))

    ideal = sizes * (n_train_total / total)
    counts = np.floor(ideal).astype(np.intp)
    remainder = n_train_total - int(counts.sum())
    if remainder > 0:
        # Hand leftover slots to the largest fractional remainders that still
        # have spare capacity (ideal <= size, so capacity always exists here).
        order = np.argsort(-(ideal - np.floor(ideal)))
        for i in order:
            if remainder == 0:
                break
            if counts[i] < sizes[i]:
                counts[i] += 1
                remainder -= 1
    return counts


def as_index_array(indices: ArrayLike) -> np.ndarray:
    """Return ``indices`` as a 1-D integer ndarray (the canonical split output)."""
    return np.asarray(list(indices), dtype=np.intp)


def compute_pairwise_distances(X: ArrayLike, metric: str = "euclidean") -> np.ndarray:
    """Compute pairwise distance matrix.

    .. warning::
        Materializes a full O(n²) matrix. For large datasets, prefer
        NearestNeighbors or chunked computation.
    """
    # TODO: Add an optional max_samples guard or return a sparse representation
    # for large inputs to avoid OOM.
    X = np.asarray(X)
    return cdist(X, X, metric=metric)


def compute_centroid(X: ArrayLike) -> np.ndarray:
    """Compute centroid of embeddings."""
    X = np.asarray(X)
    return X.mean(axis=0)


def compute_split_centroids(
    X: ArrayLike, train_indices: ArrayLike, test_indices: ArrayLike
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute centroids of train and test sets."""
    X = np.asarray(X)
    train_indices = np.asarray(train_indices, dtype=np.intp)
    test_indices = np.asarray(test_indices, dtype=np.intp)
    train_centroid = X[train_indices].mean(axis=0) if len(train_indices) else None
    test_centroid = X[test_indices].mean(axis=0) if len(test_indices) else None
    return train_centroid, test_centroid


def cluster_embeddings(
    X: ArrayLike,
    n_clusters: int = 10,
    method: str = "kmeans",
    random_state: int = 42,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[int, list[int]], np.ndarray]:
    """
    Cluster embeddings and return labels and cluster info.

    Returns:
        labels: cluster label for each sample
        cluster_to_indices: dict mapping cluster_id to list of indices
        cluster_centers: cluster centroids (for kmeans)
    """
    X = np.asarray(X)

    if method == "kmeans":
        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto",
            **kwargs
        )
        labels = clusterer.fit_predict(X)
        cluster_centers = clusterer.cluster_centers_
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    cluster_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_to_indices[label].append(idx)

    return labels, cluster_to_indices, cluster_centers


def random_split(
    embeddings: ArrayLike, train_size: float | int = 0.7, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple random train/test split (baseline).

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        random_state: int, RandomState, or None for reproducibility

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set
    """
    embeddings = validate_split_inputs(embeddings, train_size)
    n_samples = len(embeddings)
    n_train = resolve_n_train(n_samples, train_size)
    rng = check_random_state(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    return indices[:n_train], indices[n_train:]


def compute_split_similarity(
    X: ArrayLike,
    train_indices: ArrayLike,
    test_indices: ArrayLike,
    metric: str = "euclidean",
) -> dict[str, float]:
    """
    Compute similarity metrics between train and test splits.

    Returns dict with:
        - centroid_distance: distance between train/test centroids
        - mean_cross_distance: mean distance from test to nearest train
        - coverage: fraction of test samples with train neighbor within median distance
    """
    X = np.asarray(X)
    train_indices = np.asarray(train_indices, dtype=np.intp)
    test_indices = np.asarray(test_indices, dtype=np.intp)

    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError(
            "compute_split_similarity requires non-empty train and test sets"
        )

    train_X = X[train_indices]
    test_X = X[test_indices]

    # Centroid distance
    train_centroid = train_X.mean(axis=0)
    test_centroid = test_X.mean(axis=0)
    centroid_distance = np.linalg.norm(train_centroid - test_centroid)

    # Cross-set distances
    cross_distances = cdist(test_X, train_X, metric=metric)
    min_distances = cross_distances.min(axis=1)
    mean_cross_distance = min_distances.mean()

    # Coverage (fraction of test with nearby train sample)
    # TODO: Replace full pairwise matrix with sampled median estimation and
    # NearestNeighbors for coverage check to reduce O(n²) memory.
    all_distances = cdist(X, X, metric=metric)
    np.fill_diagonal(all_distances, np.inf)
    median_dist = np.median(all_distances[all_distances < np.inf])
    coverage = (min_distances <= median_dist).mean()

    return {
        "centroid_distance": float(centroid_distance),
        "mean_cross_distance": float(mean_cross_distance),
        "coverage": float(coverage),
    }


def optimized_split(
    embeddings: np.ndarray,
    train_size: float | int,
    n_iterations: int,
    score_fn: Callable[[np.ndarray, list[int], list[int]], float],
    random_state: int = 42,
    minimize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterative swap-optimization to find a split that optimizes a score.

    Starts from a random split and repeatedly swaps one train/test pair,
    keeping the swap only if the score improves.

    Args:
        embeddings: array of shape (n_samples, embedding_dim), already np.ndarray
        train_size: fraction in (0, 1) or absolute count for the training set
        n_iterations: number of swap attempts
        score_fn: callable(embeddings, train_indices, test_indices) -> float
        random_state: int, RandomState, or None for reproducibility
        minimize: if True, accept swaps that lower the score;
                  if False, accept swaps that raise it
    """
    n_samples = len(embeddings)
    rng = check_random_state(random_state)

    all_indices = np.arange(n_samples)
    rng.shuffle(all_indices)

    n_train = resolve_n_train(n_samples, train_size)
    train_indices = set(all_indices[:n_train].tolist())
    test_indices = set(all_indices[n_train:].tolist())

    current_score = score_fn(embeddings, list(train_indices), list(test_indices))

    for _ in range(n_iterations):
        train_sample = rng.choice(list(train_indices))
        test_sample = rng.choice(list(test_indices))

        # swap
        train_indices.remove(train_sample)
        train_indices.add(test_sample)
        test_indices.remove(test_sample)
        test_indices.add(train_sample)

        new_score = score_fn(embeddings, list(train_indices), list(test_indices))

        improved = new_score < current_score if minimize else new_score > current_score
        if improved:
            current_score = new_score
        else:
            # revert
            train_indices.remove(test_sample)
            train_indices.add(train_sample)
            test_indices.remove(train_sample)
            test_indices.add(test_sample)

    return as_index_array(sorted(train_indices)), as_index_array(sorted(test_indices))


def greedy_assign_to_target(
    items_with_sizes: list[tuple[int, int]], target_size: int
) -> tuple[list[int], list[int]]:
    """
    Greedily assign items to reach target size.

    Args:
        items_with_sizes: list of (item_id, size) tuples
        target_size: target total size

    Returns:
        selected: list of item_ids assigned
        remaining: list of item_ids not assigned
    """
    selected = []
    remaining = []
    current_size = 0

    for item_id, size in items_with_sizes:
        if current_size + size <= target_size:
            selected.append(item_id)
            current_size += size
        else:
            remaining.append(item_id)

    return selected, remaining
