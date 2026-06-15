"""
Split-quality reporting.

Quantifies *how* adversarial, overlapping, or balanced a produced train/test
split actually is — so users can compare splitters and so papers can report a
single, interpretable table. Builds on :func:`splytters.utils.compute_split_similarity`
and :func:`splytters.adversarial.get_cluster_info`, adding distribution-distance
metrics (MMD, energy distance, mean 1-D Wasserstein / KS) and label balance.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.utils import check_random_state

from splytters._types import Splitter
from splytters.adversarial import get_cluster_info
from splytters.utils import compute_split_similarity, validate_split_inputs


def _rbf_mmd(A: np.ndarray, B: np.ndarray, gamma: float) -> float:
    """Biased RBF Maximum Mean Discrepancy between sample sets A and B."""
    m, n = len(A), len(B)
    k_aa = np.exp(-gamma * cdist(A, A, metric="sqeuclidean"))
    k_bb = np.exp(-gamma * cdist(B, B, metric="sqeuclidean"))
    k_ab = np.exp(-gamma * cdist(A, B, metric="sqeuclidean"))
    return float(
        k_aa.sum() / (m * m) + k_bb.sum() / (n * n) - 2 * k_ab.sum() / (m * n)
    )


def _energy_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Multivariate energy distance: 2·E‖A-B‖ - E‖A-A‖ - E‖B-B‖."""
    d_ab = cdist(A, B).mean()
    d_aa = cdist(A, A).mean()
    d_bb = cdist(B, B).mean()
    return float(2 * d_ab - d_aa - d_bb)


def _subsample(
    X: np.ndarray, idx: np.ndarray, max_samples: int | None, rng
) -> np.ndarray:
    """Return X[idx], optionally subsampled to at most max_samples rows."""
    if max_samples is not None and len(idx) > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)
    return X[np.asarray(idx, dtype=np.intp)]


def split_report(
    embeddings: ArrayLike,
    train_indices: ArrayLike,
    test_indices: ArrayLike,
    y: ArrayLike | None = None,
    *,
    metric: str = "euclidean",
    n_clusters: int = 10,
    max_samples: int | None = 2000,
    random_state: int | None = 42,
) -> dict[str, float]:
    """Summarize how similar/dissimilar a train/test split is.

    Larger distribution distances (``mmd_rbf``, ``energy_distance``,
    ``wasserstein_mean``, ``ks_mean``) and ``mean_cross_distance`` indicate a
    *more adversarial* (harder) split; values near a random split indicate a
    *balanced* split; near-zero indicates *overlap*.

    Args:
        embeddings: array-like of shape (n_samples, n_features).
        train_indices: integer index array for the training split.
        test_indices: integer index array for the test split.
        y: array-like of labels, optional. If given, adds
            ``label_distribution_shift`` (total-variation distance between
            train/test class proportions; 0 = identical balance).
        metric: distance metric (default ``"euclidean"``).
        n_clusters: clusters used for the leakage statistic (default 10).
        max_samples: cap per side for the O(n²) distribution metrics
            (subsampled for speed); ``None`` uses all samples (default 2000).
        random_state: seed for subsampling (default 42).

    Returns:
        A dict of structural, geometric, distributional, and (optional) label
        metrics.
    """
    X = validate_split_inputs(embeddings, 0.5)  # reuse finite/2-D validation
    train_indices = np.asarray(train_indices, dtype=np.intp)
    test_indices = np.asarray(test_indices, dtype=np.intp)
    rng = check_random_state(random_state)

    n_train, n_test = len(train_indices), len(test_indices)
    report: dict[str, float] = {
        "n_train": float(n_train),
        "n_test": float(n_test),
        "train_fraction": float(n_train / (n_train + n_test)),
    }

    # Geometric similarity (centroid distance, nearest-train distance, coverage).
    report.update(compute_split_similarity(X, train_indices, test_indices, metric))

    # Cluster leakage.
    info = get_cluster_info(
        X, train_indices, test_indices, n_clusters=n_clusters,
        random_state=random_state if isinstance(random_state, int) else 42,
    )
    report["cluster_leakage_ratio"] = float(info["leakage_ratio"])

    # Distribution distances on (optionally subsampled) embeddings.
    A = _subsample(X, train_indices, max_samples, rng)
    B = _subsample(X, test_indices, max_samples, rng)
    gamma = 1.0 / X.shape[1]
    report["mmd_rbf"] = _rbf_mmd(A, B, gamma)
    report["energy_distance"] = _energy_distance(A, B)
    report["wasserstein_mean"] = float(
        np.mean([wasserstein_distance(A[:, d], B[:, d]) for d in range(X.shape[1])])
    )
    report["ks_mean"] = float(
        np.mean([ks_2samp(A[:, d], B[:, d]).statistic for d in range(X.shape[1])])
    )

    # Label balance: total-variation distance between class proportions.
    if y is not None:
        y = np.asarray(y)
        y_train, y_test = y[train_indices], y[test_indices]
        tv = 0.0
        for c in np.unique(y):
            p_train = (y_train == c).mean() if n_train else 0.0
            p_test = (y_test == c).mean() if n_test else 0.0
            tv += abs(p_train - p_test)
        report["label_distribution_shift"] = float(0.5 * tv)

    return report


def compare_splitters(
    embeddings: ArrayLike,
    splitters: dict[str, Splitter],
    y: ArrayLike | None = None,
    *,
    train_size: float | int = 0.7,
    random_state: int | None = 42,
    **report_kwargs: Any,
) -> dict[str, dict[str, float]]:
    """Run several splitters and return ``{name: split_report(...)}``.

    Handy for generating a comparison table (e.g. random vs adversarial vs
    balanced) in one call.
    """
    out: dict[str, dict[str, float]] = {}
    for name, splitter in splitters.items():
        train_idx, test_idx = splitter(
            embeddings, train_size=train_size, random_state=random_state
        )
        out[name] = split_report(
            embeddings, train_idx, test_idx, y, random_state=random_state,
            **report_kwargs,
        )
    return out
