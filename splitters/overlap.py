"""
High-overlap splitting algorithms that maximize train-test similarity.

These methods create "easy" evaluation sets where test samples are
similar to training samples, useful for sanity checks and debugging.
"""

from collections import defaultdict

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from splitters.utils import cluster_embeddings, compute_centroid


def cluster_leak_split(
    embeddings,
    train_ratio=0.7,
    n_clusters=10,
    random_state=42,
    **cluster_kwargs
):
    """
    Split clusters across train/test to maximize similarity.

    Instead of assigning entire clusters to one set (adversarial),
    this splits each cluster proportionally between train and test,
    ensuring similar samples appear in both sets.

    Args:
        embeddings: np.array of shape (n_samples, embedding_dim)
        train_ratio: fraction of data for training
        n_clusters: number of clusters
        random_state: for reproducibility
        **cluster_kwargs: passed to KMeans

    Returns:
        train_indices: list of indices for training set
        test_indices: list of indices for test set
    """
    embeddings = np.asarray(embeddings)
    rng = np.random.RandomState(random_state)

    labels, cluster_to_indices, _ = cluster_embeddings(
        embeddings, n_clusters, "kmeans", random_state, **cluster_kwargs
    )

    train_indices = []
    test_indices = []

    # Split each cluster proportionally
    for cluster_id, indices in cluster_to_indices.items():
        indices = np.array(indices)
        rng.shuffle(indices)

        n_train = int(len(indices) * train_ratio)
        train_indices.extend(indices[:n_train].tolist())
        test_indices.extend(indices[n_train:].tolist())

    return train_indices, test_indices


def neighbor_coverage_split(
    embeddings,
    train_ratio=0.7,
    k=5,
    metric="euclidean",
    random_state=42
):
    """
    Ensure each test sample has k similar samples in train.

    Iteratively assigns samples to test only if they have enough
    similar samples already in train, maximizing coverage.

    Args:
        embeddings: np.array of shape (n_samples, embedding_dim)
        train_ratio: fraction of data for training
        k: minimum number of similar train samples for each test sample
        metric: distance metric
        random_state: for reproducibility

    Returns:
        train_indices: list of indices for training set
        test_indices: list of indices for test set
    """
    embeddings = np.asarray(embeddings)
    n_samples = len(embeddings)
    rng = np.random.RandomState(random_state)

    # Compute pairwise distances
    distances = cdist(embeddings, embeddings, metric=metric)
    np.fill_diagonal(distances, np.inf)

    # Compute similarity threshold (median distance)
    finite_dists = distances[distances < np.inf]
    threshold = np.median(finite_dists)

    # Start with random train set
    all_indices = np.arange(n_samples)
    rng.shuffle(all_indices)

    n_train = int(n_samples * train_ratio)
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

    return list(train_set), test_indices


def centroid_matched_split(
    embeddings,
    train_ratio=0.7,
    n_iterations=100,
    random_state=42
):
    """
    Minimize distance between train and test centroids.

    Uses iterative optimization to find a split where the centroids
    of train and test sets are as close as possible.

    Args:
        embeddings: np.array of shape (n_samples, embedding_dim)
        train_ratio: fraction of data for training
        n_iterations: number of swap iterations
        random_state: for reproducibility

    Returns:
        train_indices: list of indices for training set
        test_indices: list of indices for test set
    """
    embeddings = np.asarray(embeddings)
    n_samples = len(embeddings)
    rng = np.random.RandomState(random_state)

    # Start with random split
    all_indices = np.arange(n_samples)
    rng.shuffle(all_indices)

    n_train = int(n_samples * train_ratio)
    train_indices = set(all_indices[:n_train])
    test_indices = set(all_indices[n_train:])

    def compute_centroid_distance():
        train_centroid = embeddings[list(train_indices)].mean(axis=0)
        test_centroid = embeddings[list(test_indices)].mean(axis=0)
        return np.linalg.norm(train_centroid - test_centroid)

    current_dist = compute_centroid_distance()

    # Iteratively swap samples to minimize centroid distance
    for _ in range(n_iterations):
        # Pick random samples from each set
        train_sample = rng.choice(list(train_indices))
        test_sample = rng.choice(list(test_indices))

        # Try swap
        train_indices.remove(train_sample)
        train_indices.add(test_sample)
        test_indices.remove(test_sample)
        test_indices.add(train_sample)

        new_dist = compute_centroid_distance()

        if new_dist < current_dist:
            current_dist = new_dist
        else:
            # Revert swap
            train_indices.remove(test_sample)
            train_indices.add(train_sample)
            test_indices.remove(train_sample)
            test_indices.add(test_sample)

    return list(train_indices), list(test_indices)


def stratified_similarity_split(
    embeddings,
    train_ratio=0.7,
    n_bins=10,
    random_state=42
):
    """
    Stratify by distance from centroid, ensuring similar distribution in both sets.

    Bins samples by their distance from the centroid and samples
    proportionally from each bin for both train and test.

    Args:
        embeddings: np.array of shape (n_samples, embedding_dim)
        train_ratio: fraction of data for training
        n_bins: number of distance bins for stratification
        random_state: for reproducibility

    Returns:
        train_indices: list of indices for training set
        test_indices: list of indices for test set
    """
    embeddings = np.asarray(embeddings)
    rng = np.random.RandomState(random_state)

    # Compute distances from centroid
    centroid = compute_centroid(embeddings)
    distances = np.linalg.norm(embeddings - centroid, axis=1)

    # Bin samples by distance
    bin_edges = np.percentile(distances, np.linspace(0, 100, n_bins + 1))
    bin_assignments = np.digitize(distances, bin_edges[1:-1])

    train_indices = []
    test_indices = []

    # Sample proportionally from each bin
    for bin_id in range(n_bins):
        bin_samples = np.where(bin_assignments == bin_id)[0]
        if len(bin_samples) == 0:
            continue

        rng.shuffle(bin_samples)
        n_train = max(1, int(len(bin_samples) * train_ratio))

        train_indices.extend(bin_samples[:n_train].tolist())
        test_indices.extend(bin_samples[n_train:].tolist())

    return train_indices, test_indices


def nearest_neighbor_split(
    embeddings,
    train_ratio=0.7,
    metric="euclidean",
    random_state=42
):
    """
    For each test sample, ensure its nearest neighbor is in train.

    Greedily builds test set by moving points whose nearest neighbor
    is already confirmed in train. Uses sklearn's NearestNeighbors
    which auto-selects the best algorithm (kd-tree, ball tree, or
    brute force) based on dimensionality.

    Args:
        embeddings: np.array of shape (n_samples, embedding_dim)
        train_ratio: fraction of data for training
        metric: distance metric
        random_state: for reproducibility

    Returns:
        train_indices: list of indices for training set
        test_indices: list of indices for test set
    """
    from sklearn.neighbors import NearestNeighbors

    embeddings = np.asarray(embeddings)
    n_samples = len(embeddings)
    n_test = n_samples - int(n_samples * train_ratio)
    rng = np.random.RandomState(random_state)

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

    train_indices = np.where(in_train)[0].tolist()
    return train_indices, test_indices


def duplicate_spread_split(
    embeddings,
    train_ratio=0.7,
    similarity_threshold=None,
    metric="euclidean",
    random_state=42
):
    """
    Intentionally put near-duplicates in both train and test.

    Identifies clusters of near-duplicates and ensures at least one
    sample from each cluster appears in both sets.

    Args:
        embeddings: np.array of shape (n_samples, embedding_dim)
        train_ratio: fraction of data for training
        similarity_threshold: distance threshold for near-duplicates
                              (default: 10th percentile of distances)
        metric: distance metric
        random_state: for reproducibility

    Returns:
        train_indices: list of indices for training set
        test_indices: list of indices for test set
    """
    embeddings = np.asarray(embeddings)
    n_samples = len(embeddings)
    rng = np.random.RandomState(random_state)

    # Compute pairwise distances
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

    train_indices = []
    test_indices = []

    # For each component, split proportionally (ensuring both sets get samples)
    for component_id, indices in component_to_indices.items():
        indices = np.array(indices)
        rng.shuffle(indices)

        if len(indices) == 1:
            # Single sample: assign to train
            train_indices.extend(indices.tolist())
        else:
            # Split ensuring both sets get at least one
            n_train = max(1, int(len(indices) * train_ratio))
            n_train = min(n_train, len(indices) - 1)  # Leave at least 1 for test

            train_indices.extend(indices[:n_train].tolist())
            test_indices.extend(indices[n_train:].tolist())

    return train_indices, test_indices


def max_coverage_split(
    embeddings,
    train_ratio=0.7,
    radius=None,
    metric="euclidean",
    random_state=42
):
    """
    Maximize the coverage of test set by train set.

    Coverage = fraction of test samples with at least one train
    sample within radius.

    Args:
        embeddings: np.array of shape (n_samples, embedding_dim)
        train_ratio: fraction of data for training
        radius: distance threshold for coverage (default: median distance)
        metric: distance metric
        random_state: for reproducibility

    Returns:
        train_indices: list of indices for training set
        test_indices: list of indices for test set
    """
    embeddings = np.asarray(embeddings)
    n_samples = len(embeddings)
    rng = np.random.RandomState(random_state)

    # Compute pairwise distances
    distances = cdist(embeddings, embeddings, metric=metric)
    np.fill_diagonal(distances, np.inf)

    # Set radius
    if radius is None:
        finite_dists = distances[distances < np.inf]
        radius = np.median(finite_dists)

    # Start with random split
    all_indices = np.arange(n_samples)
    rng.shuffle(all_indices)

    n_train = int(n_samples * train_ratio)
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

    return list(train_set), list(test_set)
