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


def validate_split_inputs(
    embeddings: ArrayLike, train_ratio: float, min_samples: int = 2
) -> None:
    """Validate common inputs for splitting functions.

    Raises:
        ValueError: if train_ratio is out of (0, 1) or there are too few samples.
    """
    if not 0 < train_ratio < 1:
        raise ValueError(
            f"train_ratio must be between 0 and 1 exclusive, got {train_ratio}"
        )
    n = len(np.asarray(embeddings))
    if n < min_samples:
        raise ValueError(
            f"Need at least {min_samples} samples to split, got {n}"
        )


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
    X: ArrayLike, train_indices: list[int], test_indices: list[int]
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute centroids of train and test sets."""
    X = np.asarray(X)
    train_centroid = X[train_indices].mean(axis=0) if train_indices else None
    test_centroid = X[test_indices].mean(axis=0) if test_indices else None
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
    embeddings: ArrayLike, train_ratio: float = 0.7, random_state: int = 42
) -> tuple[list[int], list[int]]:
    """
    Simple random train/test split (baseline).

    Args:
        embeddings: np.array of shape (n_samples, embedding_dim)
        train_ratio: fraction of data for training
        random_state: for reproducibility

    Returns:
        train_indices: list of indices for training set
        test_indices: list of indices for test set
    """
    validate_split_inputs(embeddings, train_ratio)
    embeddings = np.asarray(embeddings)
    n_samples = len(embeddings)
    n_train = int(n_samples * train_ratio)
    rng = np.random.RandomState(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    return indices[:n_train].tolist(), indices[n_train:].tolist()


def compute_split_similarity(
    X: ArrayLike,
    train_indices: list[int],
    test_indices: list[int],
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
        "centroid_distance": centroid_distance,
        "mean_cross_distance": mean_cross_distance,
        "coverage": coverage,
    }


def optimized_split(
    embeddings: np.ndarray,
    train_ratio: float,
    n_iterations: int,
    score_fn: Callable[[np.ndarray, list[int], list[int]], float],
    random_state: int = 42,
    minimize: bool = True,
) -> tuple[list[int], list[int]]:
    """Iterative swap-optimization to find a split that optimizes a score.

    Starts from a random split and repeatedly swaps one train/test pair,
    keeping the swap only if the score improves.

    Args:
        embeddings: array of shape (n_samples, embedding_dim), already np.ndarray
        train_ratio: fraction of data for training
        n_iterations: number of swap attempts
        score_fn: callable(embeddings, train_indices, test_indices) -> float
        random_state: for reproducibility
        minimize: if True, accept swaps that lower the score;
                  if False, accept swaps that raise it
    """
    n_samples = len(embeddings)
    rng = np.random.RandomState(random_state)

    all_indices = np.arange(n_samples)
    rng.shuffle(all_indices)

    n_train = int(n_samples * train_ratio)
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

    return list(train_indices), list(test_indices)


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
