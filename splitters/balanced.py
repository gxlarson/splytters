"""
Balanced splitting algorithms that match distributions between train/test.

These methods create splits where train and test have similar statistical
properties, useful for fair evaluation without distribution shift.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp

from splitters.utils import compute_centroid


def distribution_matched_split(
    embeddings,
    train_ratio=0.7,
    n_iterations=1000,
    random_state=42
):
    """
    Minimize distribution divergence between train and test.

    Uses iterative optimization to match the marginal distributions
    of each feature dimension.

    Args:
        embeddings: np.array of shape (n_samples, embedding_dim)
        train_ratio: fraction of data for training
        n_iterations: number of optimization iterations
        random_state: for reproducibility

    Returns:
        train_indices: list of indices for training set
        test_indices: list of indices for test set
    """
    embeddings = np.asarray(embeddings)
    n_samples, n_dims = embeddings.shape
    rng = np.random.RandomState(random_state)

    # Start with random split
    all_indices = np.arange(n_samples)
    rng.shuffle(all_indices)

    n_train = int(n_samples * train_ratio)
    train_indices = set(all_indices[:n_train])
    test_indices = set(all_indices[n_train:])

    def compute_divergence():
        """Compute mean KS statistic across all dimensions."""
        train_data = embeddings[list(train_indices)]
        test_data = embeddings[list(test_indices)]

        ks_stats = []
        for dim in range(n_dims):
            stat, _ = ks_2samp(train_data[:, dim], test_data[:, dim])
            ks_stats.append(stat)

        return np.mean(ks_stats)

    current_divergence = compute_divergence()

    # Iterative optimization
    for _ in range(n_iterations):
        # Pick random samples to swap
        train_sample = rng.choice(list(train_indices))
        test_sample = rng.choice(list(test_indices))

        # Try swap
        train_indices.remove(train_sample)
        train_indices.add(test_sample)
        test_indices.remove(test_sample)
        test_indices.add(train_sample)

        new_divergence = compute_divergence()

        if new_divergence < current_divergence:
            current_divergence = new_divergence
        else:
            # Revert swap
            train_indices.remove(test_sample)
            train_indices.add(train_sample)
            test_indices.remove(train_sample)
            test_indices.add(test_sample)

    return list(train_indices), list(test_indices)


def moment_matched_split(
    embeddings,
    train_ratio=0.7,
    n_iterations=1000,
    match_variance=True,
    random_state=42
):
    """
    Match mean (and optionally variance) between train and test.

    Args:
        embeddings: np.array of shape (n_samples, embedding_dim)
        train_ratio: fraction of data for training
        n_iterations: number of optimization iterations
        match_variance: if True, also match variance
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

    def compute_moment_diff():
        train_data = embeddings[list(train_indices)]
        test_data = embeddings[list(test_indices)]

        # Mean difference
        mean_diff = np.linalg.norm(train_data.mean(axis=0) - test_data.mean(axis=0))

        if match_variance:
            # Variance difference
            var_diff = np.linalg.norm(train_data.var(axis=0) - test_data.var(axis=0))
            return mean_diff + var_diff

        return mean_diff

    current_diff = compute_moment_diff()

    # Iterative optimization
    for _ in range(n_iterations):
        train_sample = rng.choice(list(train_indices))
        test_sample = rng.choice(list(test_indices))

        # Try swap
        train_indices.remove(train_sample)
        train_indices.add(test_sample)
        test_indices.remove(test_sample)
        test_indices.add(train_sample)

        new_diff = compute_moment_diff()

        if new_diff < current_diff:
            current_diff = new_diff
        else:
            # Revert
            train_indices.remove(test_sample)
            train_indices.add(train_sample)
            test_indices.remove(train_sample)
            test_indices.add(test_sample)

    return list(train_indices), list(test_indices)


def histogram_matched_split(
    embeddings,
    train_ratio=0.7,
    n_bins=10,
    n_iterations=1000,
    random_state=42
):
    """
    Match feature histograms between train and test.

    Minimizes the sum of histogram differences across all dimensions.

    Args:
        embeddings: np.array of shape (n_samples, embedding_dim)
        train_ratio: fraction of data for training
        n_bins: number of histogram bins per dimension
        n_iterations: number of optimization iterations
        random_state: for reproducibility

    Returns:
        train_indices: list of indices for training set
        test_indices: list of indices for test set
    """
    embeddings = np.asarray(embeddings)
    n_samples, n_dims = embeddings.shape
    rng = np.random.RandomState(random_state)

    # Compute bin edges for each dimension
    bin_edges = []
    for dim in range(n_dims):
        edges = np.percentile(embeddings[:, dim], np.linspace(0, 100, n_bins + 1))
        bin_edges.append(edges)

    # Start with random split
    all_indices = np.arange(n_samples)
    rng.shuffle(all_indices)

    n_train = int(n_samples * train_ratio)
    train_indices = set(all_indices[:n_train])
    test_indices = set(all_indices[n_train:])

    def compute_histogram_diff():
        train_data = embeddings[list(train_indices)]
        test_data = embeddings[list(test_indices)]

        total_diff = 0
        for dim in range(n_dims):
            train_hist, _ = np.histogram(train_data[:, dim], bins=bin_edges[dim], density=True)
            test_hist, _ = np.histogram(test_data[:, dim], bins=bin_edges[dim], density=True)
            total_diff += np.sum(np.abs(train_hist - test_hist))

        return total_diff

    current_diff = compute_histogram_diff()

    # Iterative optimization
    for _ in range(n_iterations):
        train_sample = rng.choice(list(train_indices))
        test_sample = rng.choice(list(test_indices))

        # Try swap
        train_indices.remove(train_sample)
        train_indices.add(test_sample)
        test_indices.remove(test_sample)
        test_indices.add(train_sample)

        new_diff = compute_histogram_diff()

        if new_diff < current_diff:
            current_diff = new_diff
        else:
            # Revert
            train_indices.remove(test_sample)
            train_indices.add(train_sample)
            test_indices.remove(train_sample)
            test_indices.add(test_sample)

    return list(train_indices), list(test_indices)


def stratified_random_split(
    embeddings,
    labels,
    train_ratio=0.7,
    random_state=42
):
    """
    Standard stratified split maintaining label proportions.

    Args:
        embeddings: np.array of shape (n_samples, embedding_dim)
        labels: array of labels for stratification
        train_ratio: fraction of data for training
        random_state: for reproducibility

    Returns:
        train_indices: list of indices for training set
        test_indices: list of indices for test set
    """
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(embeddings))

    train_indices, test_indices = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=labels,
        random_state=random_state
    )

    return train_indices.tolist(), test_indices.tolist()


def density_balanced_split(
    embeddings,
    train_ratio=0.7,
    n_bins=10,
    random_state=42
):
    """
    Balance local density distribution between train and test.

    Bins samples by local density and samples proportionally
    from each bin for both sets.

    Args:
        embeddings: np.array of shape (n_samples, embedding_dim)
        train_ratio: fraction of data for training
        n_bins: number of density bins
        random_state: for reproducibility

    Returns:
        train_indices: list of indices for training set
        test_indices: list of indices for test set
    """
    embeddings = np.asarray(embeddings)
    n_samples = len(embeddings)
    rng = np.random.RandomState(random_state)

    # Compute pairwise distances
    distances = cdist(embeddings, embeddings, metric="euclidean")
    np.fill_diagonal(distances, np.inf)

    # Compute density (inverse of mean distance to 10 nearest neighbors)
    k = min(10, n_samples - 1)
    knn_distances = np.sort(distances, axis=1)[:, :k]
    densities = 1.0 / (knn_distances.mean(axis=1) + 1e-10)

    # Bin by density
    bin_edges = np.percentile(densities, np.linspace(0, 100, n_bins + 1))
    bin_assignments = np.digitize(densities, bin_edges[1:-1])

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


def mmd_minimized_split(
    embeddings,
    train_ratio=0.7,
    n_iterations=500,
    kernel="rbf",
    gamma=None,
    random_state=42
):
    """
    Minimize Maximum Mean Discrepancy between train and test.

    MMD is a kernel-based measure of distribution difference.
    Lower MMD indicates more similar distributions.

    Args:
        embeddings: np.array of shape (n_samples, embedding_dim)
        train_ratio: fraction of data for training
        n_iterations: number of optimization iterations
        kernel: kernel type ('rbf' or 'linear')
        gamma: RBF kernel parameter (default: 1/n_features)
        random_state: for reproducibility

    Returns:
        train_indices: list of indices for training set
        test_indices: list of indices for test set
    """
    embeddings = np.asarray(embeddings)
    n_samples, n_dims = embeddings.shape
    rng = np.random.RandomState(random_state)

    if gamma is None:
        gamma = 1.0 / n_dims

    def rbf_kernel(X, Y):
        distances = cdist(X, Y, metric="sqeuclidean")
        return np.exp(-gamma * distances)

    def linear_kernel(X, Y):
        return X @ Y.T

    kernel_fn = rbf_kernel if kernel == "rbf" else linear_kernel

    # Start with random split
    all_indices = np.arange(n_samples)
    rng.shuffle(all_indices)

    n_train = int(n_samples * train_ratio)
    train_indices = set(all_indices[:n_train])
    test_indices = set(all_indices[n_train:])

    def compute_mmd():
        train_data = embeddings[list(train_indices)]
        test_data = embeddings[list(test_indices)]

        K_tt = kernel_fn(train_data, train_data)
        K_ss = kernel_fn(test_data, test_data)
        K_ts = kernel_fn(train_data, test_data)

        m, n = len(train_data), len(test_data)

        mmd = (K_tt.sum() / (m * m) +
               K_ss.sum() / (n * n) -
               2 * K_ts.sum() / (m * n))

        return mmd

    current_mmd = compute_mmd()

    # Iterative optimization
    for _ in range(n_iterations):
        train_sample = rng.choice(list(train_indices))
        test_sample = rng.choice(list(test_indices))

        # Try swap
        train_indices.remove(train_sample)
        train_indices.add(test_sample)
        test_indices.remove(test_sample)
        test_indices.add(train_sample)

        new_mmd = compute_mmd()

        if new_mmd < current_mmd:
            current_mmd = new_mmd
        else:
            # Revert
            train_indices.remove(test_sample)
            train_indices.add(train_sample)
            test_indices.remove(train_sample)
            test_indices.add(test_sample)

    return list(train_indices), list(test_indices)
