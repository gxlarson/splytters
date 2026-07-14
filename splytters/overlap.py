"""
High-overlap splitting algorithms that maximize train-test similarity.

These methods create "easy" evaluation sets where test samples are
similar to training samples, useful for sanity checks and debugging.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist
from sklearn.utils import check_random_state

from splytters.utils import (
    apportion_train,
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

    Seed stability: varies with the seed like a random split -- each cluster is
    shuffled before being split, so which members go to test differs run to run.
    """
    embeddings = validate_split_inputs(embeddings, train_size)
    rng = check_random_state(random_state)

    labels, cluster_to_indices, _ = cluster_embeddings(
        embeddings, n_clusters, "kmeans", random_state, **cluster_kwargs
    )

    n_samples = len(embeddings)
    n_train_total = resolve_n_train(n_samples, train_size)

    # Apportion the global train target across clusters (largest-remainder) so
    # the realized split matches train_size instead of undershooting from
    # per-cluster truncation.
    clusters = [np.array(idx) for idx in cluster_to_indices.values()]
    per_cluster_train = apportion_train([len(c) for c in clusters], n_train_total)

    train_indices = []
    test_indices = []
    for indices, k in zip(clusters, per_cluster_train, strict=True):
        rng.shuffle(indices)
        train_indices.extend(indices[:k].tolist())
        test_indices.extend(indices[k:].tolist())

    return as_index_array(train_indices), as_index_array(test_indices)


def neighbor_coverage_split(
    embeddings: ArrayLike,
    train_size: float | int = 0.7,
    k: int = 5,
    metric: str = "euclidean",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Best-effort: ensure each test sample has k similar samples in train.

    Solves a binary feasibility problem whose constraints require every test
    sample to have at least ``k`` train neighbors within the median pairwise
    distance. Raises when no split of the requested size is feasible. When a
    solver returns a split that does not fully meet the coverage constraint
    (off-constraint solution), the promise is best-effort: the split is still
    returned and a warning reports how many test samples are under-covered,
    rather than silently violating the documented guarantee.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        k: minimum number of similar train samples for each test sample
        metric: distance metric
        random_state: for reproducibility

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set

    Seed stability: varies with the seed. The constraints are fixed, while a
    seeded random linear objective selects among multiple feasible solutions.
    """
    embeddings = validate_split_inputs(embeddings, train_size)
    n_samples = len(embeddings)
    rng = check_random_state(random_state)

    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")

    # TODO: Replace full pairwise matrix with chunked or BallTree.query_radius
    # computation to reduce memory from O(n²). Only threshold-based neighbor
    # lookups are needed here.
    distances = cdist(embeddings, embeddings, metric=metric)
    np.fill_diagonal(distances, np.inf)

    # Compute similarity threshold (median distance)
    finite_dists = distances[distances < np.inf]
    threshold = np.median(finite_dists)

    n_train = resolve_n_train(n_samples, train_size)
    if k > n_train:
        raise ValueError(
            "no feasible neighbor-coverage split: "
            f"k={k} exceeds the {n_train} available train samples"
        )

    # Binary variable x_i is 1 when sample i is in train. For every sample i,
    #     sum_{j in N(i)} x_j + k*x_i >= k
    # activates the coverage requirement only when i is in test (x_i == 0).
    # Together with sum_i x_i == n_train this is the exact documented contract.
    from scipy import sparse
    from scipy.optimize import Bounds, LinearConstraint, milp

    neighbor_matrix = (distances <= threshold).astype(float)
    coverage_matrix = neighbor_matrix + k * np.eye(n_samples)
    constraint_matrix = sparse.vstack(
        [np.ones((1, n_samples)), sparse.csr_matrix(coverage_matrix)],
        format="csr",
    )
    lower = np.concatenate(([n_train], np.full(n_samples, float(k))))
    upper = np.concatenate(([n_train], np.full(n_samples, np.inf)))

    # The objective only breaks ties between feasible solutions. Seeded random
    # costs retain this splitter's documented seed variation.
    result = milp(
        c=rng.uniform(size=n_samples),
        integrality=np.ones(n_samples),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(constraint_matrix, lower, upper),
    )
    if not result.success or result.x is None:
        raise ValueError(
            "no feasible neighbor-coverage split for the requested train_size, "
            f"k={k}, and median-distance neighborhood graph"
        )

    train_mask = result.x >= 0.5
    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(~train_mask)
    if len(train_indices) != n_train:
        raise RuntimeError("neighbor-coverage MILP returned an invalid solution")

    # The MILP constraints enforce k-coverage, but report honestly rather than
    # silently violate the documented promise if a solver ever returns a split
    # where some test point has fewer than k train neighbors.
    coverage = neighbor_matrix[np.ix_(test_indices, train_indices)].sum(axis=1)
    n_undercovered = int(np.count_nonzero(coverage < k))
    if n_undercovered:
        warnings.warn(
            f"neighbor_coverage_split left {n_undercovered} of "
            f"{len(test_indices)} test samples with fewer than k={k} train "
            "neighbors within the median-distance neighborhood; the coverage "
            "guarantee is best-effort and was not fully met for this split.",
            stacklevel=2,
        )
    return as_index_array(train_indices), as_index_array(test_indices)


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

    Seed stability: varies with the seed like a random split -- the swap
    optimization has many assignments with equally matched centroids.
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

    Seed stability: varies with the seed like a random split -- samples are
    shuffled within each distance bin before the proportional split.
    """
    embeddings = validate_split_inputs(embeddings, train_size)
    rng = check_random_state(random_state)

    if (
        not isinstance(n_bins, (int, np.integer))
        or isinstance(n_bins, bool)
        or n_bins < 1
    ):
        raise ValueError(f"n_bins must be a positive integer, got {n_bins!r}")

    # Compute distances from centroid
    centroid = compute_centroid(embeddings)
    distances = np.linalg.norm(embeddings - centroid, axis=1)

    # Bin samples by distance
    bin_edges = np.percentile(distances, np.linspace(0, 100, n_bins + 1))
    bin_assignments = np.digitize(distances, bin_edges[1:-1])

    n_train_total = resolve_n_train(len(embeddings), train_size)

    # Collect non-empty bins, then apportion the global train target across them
    # (largest-remainder) so the split hits train_size and never empties the
    # test set, even when bins are singletons (n_bins >= n_samples).
    bins = [b for b in (np.where(bin_assignments == i)[0] for i in range(n_bins))
            if len(b) > 0]
    per_bin_train = apportion_train([len(b) for b in bins], n_train_total)

    train_indices = []
    test_indices = []
    for bin_samples, k in zip(bins, per_bin_train, strict=True):
        rng.shuffle(bin_samples)
        train_indices.extend(bin_samples[:k].tolist())
        test_indices.extend(bin_samples[k:].tolist())

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

    Seed stability: varies with the seed like a random split -- points are
    processed in a random order when building the test set.
    """
    from sklearn.neighbors import NearestNeighbors

    embeddings = validate_split_inputs(embeddings, train_size)
    n_samples = len(embeddings)
    n_test = n_samples - resolve_n_train(n_samples, train_size)
    rng = check_random_state(random_state)

    # Find each point's nearest neighbor. Query k=2 because the point itself is
    # among its own neighbors on the fitted data. Normally self is column 0
    # (distance 0), but with exact duplicates the sorted result can list the
    # point itself in either column; take whichever column is not the point's
    # own index so the recorded nearest neighbor is always another point. This
    # avoids self-neighbor no-ops that wasted test capacity and raised spurious
    # "could only place N" warnings.
    nn_model = NearestNeighbors(n_neighbors=2, metric=metric, algorithm="auto")
    nn_model.fit(embeddings)
    two_nn = nn_model.kneighbors(embeddings, return_distance=False)
    row = np.arange(n_samples)
    neighbors = np.where(two_nn[:, 0] == row, two_nn[:, 1], two_nn[:, 0])

    # Start with all points in train, then greedily move points to test.
    # A point can become test only if its NN is (and stays) in train.
    in_train = np.ones(n_samples, dtype=bool)
    # `pinned` marks train points that some test point relies on as its nearest
    # neighbor; they must never move to test, or that test point's invariant
    # would break. Without this, a later iteration could pull a point's NN into
    # test, silently violating the documented guarantee.
    pinned = np.zeros(n_samples, dtype=bool)

    # Process in random order so the result isn't biased by index ordering
    order = np.arange(n_samples)
    rng.shuffle(order)

    test_indices = []
    for idx in order:
        if len(test_indices) >= n_test:
            break
        if pinned[idx]:
            continue  # must stay in train to satisfy a dependent test point
        nn = neighbors[idx]
        if in_train[nn]:
            in_train[idx] = False
            pinned[nn] = True  # lock idx's NN into train
            test_indices.append(idx)

    if len(test_indices) < n_test:
        warnings.warn(
            f"nearest_neighbor_split could only place {len(test_indices)} of the "
            f"requested {n_test} test samples while keeping every test point's "
            "nearest neighbor in train; the test set is smaller than train_size "
            "implies. Use a larger dataset or a smaller test fraction.",
            stacklevel=2,
        )

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

    Seed stability: varies with the seed like a random split -- each
    near-duplicate group is shuffled before being split across both sides.
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

    n_train_total = resolve_n_train(len(embeddings), train_size)
    train_fraction = n_train_total / len(embeddings)

    train_indices = []
    test_indices = []
    singletons: list[int] = []

    # For each near-duplicate group of size >= 2, split it across both sides so
    # the duplicates appear in train *and* test (the point of this splitter).
    for indices in component_to_indices.values():
        indices = np.array(indices)
        rng.shuffle(indices)

        if len(indices) == 1:
            singletons.append(int(indices[0]))
            continue

        # Split ensuring both sets get at least one
        n_train = max(1, int(len(indices) * train_fraction))
        n_train = min(n_train, len(indices) - 1)  # Leave at least 1 for test

        train_indices.extend(indices[:n_train].tolist())
        test_indices.extend(indices[n_train:].tolist())

    # Singletons have no near-duplicate to spread, so where they land doesn't
    # affect the duplicates-on-both-sides property. Distribute them to bring the
    # realized split close to train_size, instead of dumping them all in train
    # (which skewed the split and could starve the test set).
    rng.shuffle(singletons)
    n_needed = max(0, n_train_total - len(train_indices))
    train_indices.extend(singletons[:n_needed])
    test_indices.extend(singletons[n_needed:])

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

    Seed stability: varies with the seed like a random split -- it starts from a
    random split and greedily swaps, reaching different high-coverage solutions.
    """
    embeddings = validate_split_inputs(embeddings, train_size)
    n_samples = len(embeddings)
    rng = check_random_state(random_state)

    # TODO: Replace full pairwise matrix with BallTree.query_radius to check
    # coverage without materializing O(n²) distances.
    distances = cdist(embeddings, embeddings, metric=metric)
    np.fill_diagonal(distances, np.inf)  # a point never covers itself

    # Set radius
    if radius is None:
        finite_dists = distances[distances < np.inf]
        radius = np.median(finite_dists)

    # Boolean "within radius" graph, plus a running count of how many *train*
    # points cover each sample. Maintaining cover_count incrementally makes each
    # candidate swap an O(n) update rather than an O(n²) full recompute of
    # coverage (the previous nested-loop version was O(n⁴) worst case).
    within = distances <= radius

    # Start with a random split.
    all_indices = np.arange(n_samples)
    rng.shuffle(all_indices)
    n_train = resolve_n_train(n_samples, train_size)
    train_mask = np.zeros(n_samples, dtype=bool)
    train_mask[all_indices[:n_train]] = True

    cover_count = within[:, train_mask].sum(axis=1)  # train points within radius

    def coverage(test_mask: np.ndarray, cc: np.ndarray) -> float:
        return float((cc[test_mask] > 0).mean()) if test_mask.any() else 1.0

    current_coverage = coverage(~train_mask, cover_count)

    # Greedy optimization: repeatedly take an uncovered test point and, if some
    # train point can swap in to raise overall coverage, apply the best such swap.
    max_iterations = n_samples * 2
    for _ in range(max_iterations):
        test_mask = ~train_mask
        uncovered = np.flatnonzero(test_mask & (cover_count == 0))
        if uncovered.size == 0:
            break  # every test point is covered; nothing left to improve

        improved = False
        train_idx = np.flatnonzero(train_mask)
        for u in uncovered:
            base = cover_count + within[:, u]  # effect of moving u into train
            best_v, best_cov = None, current_coverage
            for v in train_idx:
                new_cc = base - within[:, v]  # and moving v out of train
                new_test = test_mask.copy()
                new_test[u] = False
                new_test[v] = True
                cov = coverage(new_test, new_cc)
                if cov > best_cov:
                    best_cov, best_v = cov, v
            if best_v is not None:
                cover_count = cover_count + within[:, u] - within[:, best_v]
                train_mask[u] = True
                train_mask[best_v] = False
                current_coverage = best_cov
                improved = True
                break
        if not improved:
            break

    return (
        as_index_array(np.flatnonzero(train_mask)),
        as_index_array(np.flatnonzero(~train_mask)),
    )
