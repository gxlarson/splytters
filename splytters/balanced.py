"""
Balanced splitting algorithms that match distributions between train/test.

These methods create splits where train and test have similar statistical
properties, useful for fair evaluation without distribution shift.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp
from sklearn.utils import check_random_state

from splytters.utils import (
    apportion_train,
    as_index_array,
    optimized_split,
    resolve_n_train,
    validate_split_inputs,
)


def distribution_matched_split(
    embeddings: ArrayLike,
    train_size: float | int = 0.7,
    n_iterations: int = 1000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimize distribution divergence between train and test.

    Uses iterative optimization to match the marginal distributions
    of each feature dimension.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        n_iterations: number of optimization iterations
        random_state: for reproducibility

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set
    """
    embeddings = validate_split_inputs(embeddings, train_size)

    def score_fn(X: np.ndarray, train: list[int], test: list[int]) -> float:
        """Mean KS statistic across all dimensions."""
        train_data, test_data = X[train], X[test]
        return float(np.mean([
            ks_2samp(train_data[:, d], test_data[:, d]).statistic
            for d in range(X.shape[1])
        ]))

    return optimized_split(
        embeddings, train_size, n_iterations, score_fn, random_state
    )


def moment_matched_split(
    embeddings: ArrayLike,
    train_size: float | int = 0.7,
    n_iterations: int = 1000,
    match_variance: bool = True,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match mean (and optionally variance) between train and test.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        n_iterations: number of optimization iterations
        match_variance: if True, also match variance
        random_state: for reproducibility

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set
    """
    embeddings = validate_split_inputs(embeddings, train_size)

    def score_fn(X: np.ndarray, train: list[int], test: list[int]) -> float:
        train_data, test_data = X[train], X[test]
        mean_diff = np.linalg.norm(train_data.mean(axis=0) - test_data.mean(axis=0))
        if match_variance:
            var_diff = np.linalg.norm(train_data.var(axis=0) - test_data.var(axis=0))
            return mean_diff + var_diff
        return mean_diff

    return optimized_split(
        embeddings, train_size, n_iterations, score_fn, random_state
    )


def histogram_matched_split(
    embeddings: ArrayLike,
    train_size: float | int = 0.7,
    n_bins: int = 10,
    n_iterations: int = 1000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match feature histograms between train and test.

    Minimizes the sum of histogram differences across all dimensions.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        n_bins: number of histogram bins per dimension
        n_iterations: number of optimization iterations
        random_state: for reproducibility

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set
    """
    embeddings = validate_split_inputs(embeddings, train_size)
    n_dims = embeddings.shape[1]

    # Precompute bin edges for each dimension. Skip degenerate dimensions whose
    # percentile edges collapse (constant/near-constant features) — a zero-width
    # bin makes density=True divide by zero and poison the score with NaN, which
    # would silently stall the optimizer (every NaN comparison is False).
    bin_edges = [
        np.percentile(embeddings[:, d], np.linspace(0, 100, n_bins + 1))
        for d in range(n_dims)
    ]
    valid_dims = [d for d in range(n_dims) if np.all(np.diff(bin_edges[d]) > 0)]

    def score_fn(X: np.ndarray, train: list[int], test: list[int]) -> float:
        train_data, test_data = X[train], X[test]
        total_diff = 0.0
        for d in valid_dims:
            train_hist, _ = np.histogram(train_data[:, d], bins=bin_edges[d], density=True)
            test_hist, _ = np.histogram(test_data[:, d], bins=bin_edges[d], density=True)
            total_diff += np.sum(np.abs(train_hist - test_hist))
        return total_diff

    return optimized_split(
        embeddings, train_size, n_iterations, score_fn, random_state
    )


def stratified_random_split(
    embeddings: ArrayLike,
    y: ArrayLike,
    train_size: float | int = 0.7,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standard stratified split maintaining label proportions.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        y: array of labels for stratification (aligned with sklearn's ``split(X, y)``)
        train_size: fraction in (0, 1) or absolute count for the training set
        random_state: for reproducibility

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set
    """
    from sklearn.model_selection import train_test_split

    embeddings = validate_split_inputs(embeddings, train_size)
    indices = np.arange(len(embeddings))

    train_indices, test_indices = train_test_split(
        indices,
        train_size=train_size,
        stratify=y,
        random_state=random_state
    )

    return as_index_array(train_indices), as_index_array(test_indices)


def density_balanced_split(
    embeddings: ArrayLike,
    train_size: float | int = 0.7,
    n_bins: int = 10,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Balance local density distribution between train and test.

    Bins samples by local density and samples proportionally
    from each bin for both sets.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        n_bins: number of density bins
        random_state: for reproducibility

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set
    """
    embeddings = validate_split_inputs(embeddings, train_size)
    n_samples = len(embeddings)
    rng = check_random_state(random_state)

    # TODO: Replace full pairwise matrix with NearestNeighbors(k) to reduce
    # memory from O(n²) to O(nk). Only k distances per point are needed here.
    distances = cdist(embeddings, embeddings, metric="euclidean")
    np.fill_diagonal(distances, np.inf)

    # Compute density (inverse of mean distance to 10 nearest neighbors)
    k = min(10, n_samples - 1)
    knn_distances = np.sort(distances, axis=1)[:, :k]
    densities = 1.0 / (knn_distances.mean(axis=1) + 1e-10)

    # Bin by density
    bin_edges = np.percentile(densities, np.linspace(0, 100, n_bins + 1))
    bin_assignments = np.digitize(densities, bin_edges[1:-1])

    n_train_total = resolve_n_train(n_samples, train_size)

    # Apportion the global train target across non-empty density bins
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


def mmd_minimized_split(
    embeddings: ArrayLike,
    train_size: float | int = 0.7,
    n_iterations: int = 500,
    kernel: str = "rbf",
    gamma: float | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimize Maximum Mean Discrepancy between train and test.

    MMD is a kernel-based measure of distribution difference.
    Lower MMD indicates more similar distributions.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        n_iterations: number of optimization iterations
        kernel: kernel type ('rbf' or 'linear')
        gamma: RBF kernel parameter (default: 1/n_features)
        random_state: for reproducibility

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set

    References:
        The adversarial dual -- *maximizing* train/validation MMD for robust
        model selection under domain shift -- is studied by Napoli & White
        (2025), "Clustering-Based Validation Splits for Model Selection under
        Domain Shift," TMLR (https://openreview.net/forum?id=Q692C0WtiD). See
        :func:`splytters.mmd_maximized_split`.
    """
    embeddings = validate_split_inputs(embeddings, train_size)
    n_dims = embeddings.shape[1]

    _gamma = gamma if gamma is not None else 1.0 / n_dims

    def _rbf_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return np.exp(-_gamma * cdist(X, Y, metric="sqeuclidean"))

    def _linear_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return X @ Y.T

    kernel_fn = _rbf_kernel if kernel == "rbf" else _linear_kernel

    def score_fn(X: np.ndarray, train: list[int], test: list[int]) -> float:
        train_data, test_data = X[train], X[test]
        K_tt = kernel_fn(train_data, train_data)
        K_ss = kernel_fn(test_data, test_data)
        K_ts = kernel_fn(train_data, test_data)
        m, n = len(train_data), len(test_data)
        return (K_tt.sum() / (m * m) +
                K_ss.sum() / (n * n) -
                2 * K_ts.sum() / (m * n))

    return optimized_split(
        embeddings, train_size, n_iterations, score_fn, random_state
    )
