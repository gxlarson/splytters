"""
High-overlap splitting algorithms that maximize train-test similarity.

These methods create "easy" evaluation sets where test samples are
similar to training samples, useful for sanity checks and debugging.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist
from sklearn.utils import check_random_state

from splitters.utils import (
    as_index_array,
    cluster_embeddings,
    compute_centroid,
    optimized_split,
    resolve_n_train,
    validate_split_inputs,
)


def cluster_leak_split(
    embeddings: ArrayLike,
    train_size: float | int = 0.7,
    n_clusters: int = 10,
    random_state: int = 42,
    **cluster_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split clusters across train/test to maximize similarity.

    Instead of assigning entire clusters to one set (adversarial),
    this splits each cluster proportionally between train and test,
    ensuring similar samples appear in both sets.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        n_clusters: number of clusters
        random_state: for reproducibility
        **cluster_kwargs: passed to KMeans

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set
    """
    embeddings = validate_split_inputs(embeddings, train_size)
    rng = check_random_state(random_state)

    labels, cluster_to_indices, _ = cluster_embeddings(
        embeddings, n_clusters, "kmeans", random_state, **cluster_kwargs
    )

    train_indices = []
    test_indices = []

    # Determine the per-cluster fraction to send to train.
    n_samples = len(embeddings)
    train_fraction = resolve_n_train(n_samples, train_size) / n_samples

    # Split each cluster proportionally
    for indices in cluster_to_indices.values():
        indices = np.array(indices)
        rng.shuffle(indices)

        n_train = int(len(indices) * train_fraction)
        train_indices.extend(indices[:n_train].tolist())
        test_indices.extend(indices[n_train:].tolist())

    return as_index_array(train_indices), as_index_array(test_indices)


def neighbor_coverage_split(
    embeddings: ArrayLike,
    train_size: float | int = 0.7,
    k: int = 5,
    metric: str = "euclidean",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensure each test sample has k similar samples in train.

    Iteratively assigns samples to test only if they have enough
    similar samples already in train, maximizing coverage.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        k: minimum number of similar train samples for each test sample
        metric: distance metric
        random_state: for reproducibility

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set
    """
    embeddings = validate_split_inputs(embeddings, train_size)
    n_samples = len(embeddings)
    rng = check_random_state(random_state)

    # TODO: Replace full pairwise matrix with chunked or BallTree.query_radius
    # computation to reduce memory from O(n²). Only threshold-based neighbor
    # lookups are needed here.
    distances = cdist(embeddings, embeddings, metric=metric)
    np.fill_diagonal(distances, np.inf)

    # Compute similarity threshold (median distance)
    finite_dists = distances[distances < np.inf]
    threshold = np.median(finite_dists)

    # Start with random train set
    all_indices = np.arange(n_samples)
    rng.shuffle(all_indices)

    n_train = resolve_n_train(n_samples, train_size)
    train_set = set(all_indices[:n_train])
    remaining = list(all_indices[n_train:])

    # Iteratively improve: swap samples to maximize coverage
    test_indices = []

    for idx in remaining:
        # Count similar samples in train
        similar_in_train = sum(
            1 for t_idx in train_set
            if distances[idx, t_idx] <= threshold
        )

        if similar_in_train >= k:
            test_indices.append(idx)
        else:
            # Find a train sample to swap
            # Prefer train samples that have many similar train neighbors
            train_list = list(train_set)
            swap_candidate = None
            max_redundancy = -1

            for t_idx in train_list:
                redundancy = sum(
                    1 for other in train_set
                    if other != t_idx and distances[t_idx, other] <= threshold
                )
                if redundancy > max_redundancy:
                    max_redundancy = redundancy
                    swap_candidate = t_idx

            if swap_candidate is not None and max_redundancy > k:
                train_set.remove(swap_candidate)
                train_set.add(idx)
                test_indices.append(swap_candidate)
            else:
                test_indices.append(idx)

    return as_index_array(train_set), as_index_array(test_indices)


def centroid_matched_split(
    embeddings: ArrayLike,
    train_size: float | int = 0.7,
    n_iterations: int = 100,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimize distance between train and test centroids.

    Uses iterative optimization to find a split where the centroids
    of train and test sets are as close as possible.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        n_iterations: number of swap iterations
        random_state: for reproducibility

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set
    """
    embeddings = validate_split_inputs(embeddings, train_size)

    def score_fn(X: np.ndarray, train: list[int], test: list[int]) -> float:
        train_centroid = X[train].mean(axis=0)
        test_centroid = X[test].mean(axis=0)
        return float(np.linalg.norm(train_centroid - test_centroid))

    return optimized_split(
        embeddings, train_size, n_iterations, score_fn, random_state
    )


def stratified_similarity_split(
    embeddings: ArrayLike,
    train_size: float | int = 0.7,
    n_bins: int = 10,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stratify by distance from centroid, ensuring similar distribution in both sets.

    Bins samples by their distance from the centroid and samples
    proportionally from each bin for both train and test.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        n_bins: number of distance bins for stratification
        random_state: for reproducibility

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set
    """
    embeddings = validate_split_inputs(embeddings, train_size)
    rng = check_random_state(random_state)

    # Compute distances from centroid
    centroid = compute_centroid(embeddings)
    distances = np.linalg.norm(embeddings - centroid, axis=1)

    # Bin samples by distance
    bin_edges = np.percentile(distances, np.linspace(0, 100, n_bins + 1))
    bin_assignments = np.digitize(distances, bin_edges[1:-1])

    train_fraction = resolve_n_train(len(embeddings), train_size) / len(embeddings)

    train_indices = []
    test_indices = []

    # Sample proportionally from each bin
    for bin_id in range(n_bins):
        bin_samples = np.where(bin_assignments == bin_id)[0]
        if len(bin_samples) == 0:
            continue

        rng.shuffle(bin_samples)
        n_train = max(1, int(len(bin_samples) * train_fraction))

        train_indices.extend(bin_samples[:n_train].tolist())
        test_indices.extend(bin_samples[n_train:].tolist())

    return as_index_array(train_indices), as_index_array(test_indices)


def nearest_neighbor_split(
    embeddings: ArrayLike,
    train_size: float | int = 0.7,
    metric: str = "euclidean",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each test sample, ensure its nearest neighbor is in train.

    Greedily builds test set by moving points whose nearest neighbor
    is already confirmed in train. Uses sklearn's NearestNeighbors
    which auto-selects the best algorithm (kd-tree, ball tree, or
    brute force) based on dimensionality.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        metric: distance metric
        random_state: for reproducibility

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set
    """
    from sklearn.neighbors import NearestNeighbors

    embeddings = validate_split_inputs(embeddings, train_size)
    n_samples = len(embeddings)
    n_test = n_samples - resolve_n_train(n_samples, train_size)
    rng = check_random_state(random_state)

    # Find each point's nearest neighbor
    nn_model = NearestNeighbors(n_neighbors=2, metric=metric, algorithm="auto")
    nn_model.fit(embeddings)
    # k=2 because the first neighbor is the point itself when querying the
    # same dataset — but fit/kneighbors on the same data excludes self only
    # if we use radius; instead just grab second neighbor
    neighbors = nn_model.kneighbors(embeddings, return_distance=False)[:, 1]

    # Start with all points in train, then greedily move points to test.
    # A point can become test only if its NN is staying in train.
    in_train = np.ones(n_samples, dtype=bool)

    # Process in random order so the result isn't biased by index ordering
    order = np.arange(n_samples)
    rng.shuffle(order)

    test_indices = []
    for idx in order:
        if len(test_indices) >= n_test:
            break
        nn = neighbors[idx]
        if in_train[nn]:
            in_train[idx] = False
            test_indices.append(idx)

    train_indices = np.where(in_train)[0]
    return as_index_array(train_indices), as_index_array(test_indices)


def duplicate_spread_split(
    embeddings: ArrayLike,
    train_size: float | int = 0.7,
    similarity_threshold: float | None = None,
    metric: str = "euclidean",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Intentionally put near-duplicates in both train and test.

    Identifies clusters of near-duplicates and ensures at least one
    sample from each cluster appears in both sets.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        similarity_threshold: distance threshold for near-duplicates
                              (default: 10th percentile of distances)
        metric: distance metric
        random_state: for reproducibility

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set
    """
    embeddings = validate_split_inputs(embeddings, train_size)
    rng = check_random_state(random_state)

    # TODO: Replace full pairwise matrix with BallTree.query_radius to find
    # near-duplicates without materializing O(n²) distances.
    distances = cdist(embeddings, embeddings, metric=metric)
    np.fill_diagonal(distances, np.inf)

    # Set threshold
    if similarity_threshold is None:
        finite_dists = distances[distances < np.inf]
        similarity_threshold = np.percentile(finite_dists, 10)

    # Find near-duplicate groups using connected components
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    adjacency = (distances <= similarity_threshold).astype(int)
    n_components, labels = connected_components(csr_matrix(adjacency))

    # Group samples by component
    component_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        component_to_indices[label].append(idx)

    train_fraction = resolve_n_train(len(embeddings), train_size) / len(embeddings)

    train_indices = []
    test_indices = []

    # For each component, split proportionally (ensuring both sets get samples)
    for indices in component_to_indices.values():
        indices = np.array(indices)
        rng.shuffle(indices)

        if len(indices) == 1:
            # Single sample: assign to train
            train_indices.extend(indices.tolist())
        else:
            # Split ensuring both sets get at least one
            n_train = max(1, int(len(indices) * train_fraction))
            n_train = min(n_train, len(indices) - 1)  # Leave at least 1 for test

            train_indices.extend(indices[:n_train].tolist())
            test_indices.extend(indices[n_train:].tolist())

    return as_index_array(train_indices), as_index_array(test_indices)


def max_coverage_split(
    embeddings: ArrayLike,
    train_size: float | int = 0.7,
    radius: float | None = None,
    metric: str = "euclidean",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Maximize the coverage of test set by train set.

    Coverage = fraction of test samples with at least one train
    sample within radius.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        radius: distance threshold for coverage (default: median distance)
        metric: distance metric
        random_state: for reproducibility

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set
    """
    embeddings = validate_split_inputs(embeddings, train_size)
    n_samples = len(embeddings)
    rng = check_random_state(random_state)

    # TODO: Replace full pairwise matrix with BallTree.query_radius to check
    # coverage without materializing O(n²) distances.
    distances = cdist(embeddings, embeddings, metric=metric)
    np.fill_diagonal(distances, np.inf)

    # Set radius
    if radius is None:
        finite_dists = distances[distances < np.inf]
        radius = np.median(finite_dists)

    # Start with random split
    all_indices = np.arange(n_samples)
    rng.shuffle(all_indices)

    n_train = resolve_n_train(n_samples, train_size)
    train_set = set(all_indices[:n_train])
    test_set = set(all_indices[n_train:])

    def compute_coverage():
        covered = 0
        for t_idx in test_set:
            for tr_idx in train_set:
                if distances[t_idx, tr_idx] <= radius:
                    covered += 1
                    break
        return covered / len(test_set) if test_set else 1.0

    current_coverage = compute_coverage()

    # Greedy optimization
    improved = True
    max_iterations = n_samples * 2

    for _ in range(max_iterations):
        if not improved:
            break
        improved = False

        for test_idx in list(test_set):
            # Check if covered
            is_covered = any(
                distances[test_idx, tr_idx] <= radius
                for tr_idx in train_set
            )

            if not is_covered:
                # Find best train sample to swap
                best_swap = None
                best_coverage = current_coverage

                for train_idx in list(train_set):
                    # Simulate swap
                    train_set.remove(train_idx)
                    train_set.add(test_idx)
                    test_set.remove(test_idx)
                    test_set.add(train_idx)

                    new_coverage = compute_coverage()

                    # Revert
                    train_set.remove(test_idx)
                    train_set.add(train_idx)
                    test_set.remove(train_idx)
                    test_set.add(test_idx)

                    if new_coverage > best_coverage:
                        best_coverage = new_coverage
                        best_swap = train_idx

                if best_swap is not None:
                    train_set.remove(best_swap)
                    train_set.add(test_idx)
                    test_set.remove(test_idx)
                    test_set.add(best_swap)
                    current_coverage = best_coverage
                    improved = True
                    break

    return as_index_array(train_set), as_index_array(test_set)
