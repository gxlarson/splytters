"""
Balanced splitting algorithms that match distributions between train/test.

These methods create splits where train and test have similar statistical
properties, useful for fair evaluation without distribution shift.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import ks_2samp
from sklearn.utils import check_random_state

from splytters.utils import (
    apportion_train,
    as_index_array,
    constrained_kernel_kmeans_split,
    kneighbors_excluding_self,
    optimized_mmd_split,
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

    Seed stability: varies with the seed like a random split -- the swap
    optimization starts from a random split and many assignments match the
    distribution equally well.
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

    Seed stability: varies with the seed like a random split -- the swap
    optimization reaches different assignments with equally matched moments.
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

    Seed stability: varies with the seed like a random split -- the swap
    optimization reaches different assignments with equally matched histograms.
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

    Seed stability: varies with the seed like a random split -- it is a seeded
    stratified random draw.
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

    Seed stability: varies with the seed like a random split -- samples are
    shuffled within each density bin before the proportional split.
    """
    embeddings = validate_split_inputs(embeddings, train_size)
    n_samples = len(embeddings)
    rng = check_random_state(random_state)

    if (
        not isinstance(n_bins, (int, np.integer))
        or isinstance(n_bins, bool)
        or n_bins < 1
    ):
        raise ValueError(f"n_bins must be a positive integer, got {n_bins!r}")

    # Compute density (inverse of mean distance to 10 nearest neighbors)
    k = min(10, n_samples - 1)
    knn_distances, _ = kneighbors_excluding_self(embeddings, k)
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
    method: str = "swap",
    y: ArrayLike | None = None,
    groups: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimize Maximum Mean Discrepancy between train and test.

    MMD is a kernel-based measure of distribution difference.
    Lower MMD indicates more similar distributions.

    Two methods are available:

    - ``method="swap"`` (default): the historical swap-optimized approximation
      that lowers the MMD by accepting train/test swaps that reduce it.
    - ``method="kernel_kmeans"``: the min-MMD dual of the constrained kernel
      k-means (``k = 2``) method of Napoli & White (see References). The
      assignment step is a linear program (LP); it maximizes the kernel scatter
      (anti-clustering) so the two sides resemble each other. Pass optional ``y``
      and/or ``groups`` to enforce per-label / per-group distribution
      constraints; the size constraint comes from ``train_size``. Note: while
      the resulting MMD is far below a random split's, in practice it is
      typically around 2x the MMD reached by the default swap optimizer, so
      prefer ``method="swap"`` when the absolute lowest MMD matters and
      ``method="kernel_kmeans"`` when the label/group constraints do.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        n_iterations: swap attempts (``method="swap"``) or maximum Lloyd-style
            iterations (``method="kernel_kmeans"``)
        kernel: kernel type ('rbf' or 'linear')
        gamma: RBF kernel parameter (default: 1/n_features)
        random_state: for reproducibility
        method: ``"swap"`` (default) or ``"kernel_kmeans"``
        y: optional labels; only valid with ``method="kernel_kmeans"``, where it
            constrains per-label proportions in train and test
        groups: optional group ids; only valid with ``method="kernel_kmeans"``,
            where it constrains per-group proportions in train and test

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set

    References:
        Napoli & White (TMLR 2025; arXiv 2024), "Clustering-Based Validation
        Splits for Model Selection under Domain Shift"
        (https://openreview.net/forum?id=Q692C0WtiD) study the adversarial dual
        -- *maximizing* train/validation MMD -- via constrained kernel k-means
        with ``k = 2`` solved by linear programming (their Algorithm 1). This
        splitter minimizes MMD instead. Under ``method="kernel_kmeans"`` it uses
        the same constrained-LP kernel k-means machinery with the objective sign
        flipped (maximizing the ``k = 2`` scatter, the min-MMD dual). The default
        ``method="swap"`` remains a swap-optimized approximation of the same
        objective, matching :func:`splytters.mmd_maximized_split`. Their Nyström
        scaling for very large ``n`` is not implemented (out of scope).

    Seed stability: with ``method="swap"``, varies with the seed like a random
    split -- the swap optimization reaches different assignments with similarly
    low MMD. With ``method="kernel_kmeans"`` the result is deterministic given
    ``random_state`` (which seeds only the initial partition).
    """
    embeddings = validate_split_inputs(embeddings, train_size)
    if kernel not in {"rbf", "linear"}:
        raise ValueError(f"kernel must be 'rbf' or 'linear', got {kernel!r}")
    if method not in {"swap", "kernel_kmeans"}:
        raise ValueError(
            f"method must be 'swap' or 'kernel_kmeans', got {method!r}"
        )
    if method == "swap" and (y is not None or groups is not None):
        raise ValueError(
            "y and groups are only supported with method='kernel_kmeans'"
        )
    if method == "kernel_kmeans":
        return constrained_kernel_kmeans_split(
            embeddings,
            train_size,
            kernel=kernel,
            gamma=gamma,
            y=y,
            groups=groups,
            random_state=random_state,
            n_iterations=n_iterations,
            maximize_mmd=False,
        )

    return optimized_mmd_split(
        embeddings,
        train_size,
        n_iterations,
        kernel=kernel,
        gamma=gamma,
        random_state=random_state,
        minimize=True,
    )
