"""
Split-quality reporting.

Quantifies *how* adversarial, overlapping, or balanced a produced train/test
split actually is — so users can compare splitters and so papers can report a
single, interpretable table. Builds on :func:`splytters.utils.compute_split_similarity`
and :func:`splytters.adversarial.get_cluster_info`, adding distribution-distance
metrics (MMD, energy, mean 1-D and sliced Wasserstein, KS, Fréchet), a
classifier two-sample test (``c2st_auc``), k-NN manifold precision/recall, and
label balance.
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
from splytters.metrics import diversity_text, mean_dist
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


# Defaults for the projection / manifold metrics, kept internal so the
# split_report signature stays small.
_SLICED_WASSERSTEIN_PROJECTIONS = 50
_MANIFOLD_K = 5


def _sliced_wasserstein(A: np.ndarray, B: np.ndarray, n_projections: int, rng) -> float:
    """Mean 1-D Wasserstein over random projections.

    Unlike the per-axis ``wasserstein_mean``, random directions capture
    cross-dimensional structure of the two distributions.
    """
    dirs = rng.standard_normal((n_projections, A.shape[1]))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return float(np.mean([wasserstein_distance(A @ w, B @ w) for w in dirs]))


def _frechet_distance(A: np.ndarray, B: np.ndarray, eps: float = 1e-6) -> float:
    """Fréchet (FID-style) distance between Gaussians fit to A and B.

    ``‖μ_A − μ_B‖² + Tr(Σ_A + Σ_B − 2 (Σ_A Σ_B)^½)`` — a parametric (mean +
    covariance) distribution distance. Covariances are ``eps``-regularized for
    numerical stability; the estimate is unreliable when n_features exceeds the
    (subsampled) sample size, since the covariance is then singular.
    """
    from scipy.linalg import sqrtm

    d = A.shape[1]
    offset = eps * np.eye(d)
    cov_a = np.atleast_2d(np.cov(A, rowvar=False)) + offset
    cov_b = np.atleast_2d(np.cov(B, rowvar=False)) + offset
    covmean = sqrtm(cov_a @ cov_b)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = A.mean(axis=0) - B.mean(axis=0)
    return float(diff @ diff + np.trace(cov_a + cov_b - 2 * covmean))


def _c2st_auc(A: np.ndarray, B: np.ndarray, random_state: int) -> float:
    """Classifier two-sample test: out-of-fold AUC distinguishing A from B.

    ~0.5 means a classifier can't tell train from test (similar / overlap);
    →1.0 means they are trivially separable (dissimilar / adversarial). AUC is
    used (not accuracy) so unequal train/test sizes don't bias the score. A
    random forest is used (not a linear model) so the test also detects
    *non-linear* shifts — e.g. a radial shell-vs-core split, which a linear
    classifier, and the mean-based distance metrics, would miss.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    n_folds = min(5, len(A), len(B))
    if n_folds < 2:
        return float("nan")  # a side too small to cross-validate
    X = np.vstack([A, B])
    y = np.concatenate([np.zeros(len(A)), np.ones(len(B))])
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def _manifold_precision_recall(
    A: np.ndarray, B: np.ndarray, k: int
) -> tuple[float, float]:
    """k-NN manifold precision/recall (Kynkäänniemi et al., 2019).

    ``precision`` = fraction of B (test) inside the train manifold — the union
    of each train point's k-NN ball; ``recall`` = fraction of A (train) inside
    the test manifold. High precision means test is well supported by train; low
    recall means test misses parts of the train support.
    """
    from sklearn.neighbors import NearestNeighbors

    k_a, k_b = min(k, len(A) - 1), min(k, len(B) - 1)
    if k_a < 1 or k_b < 1:
        return float("nan"), float("nan")
    radii_a = NearestNeighbors(n_neighbors=k_a + 1).fit(A).kneighbors(A)[0][:, -1]
    radii_b = NearestNeighbors(n_neighbors=k_b + 1).fit(B).kneighbors(B)[0][:, -1]
    precision = float((cdist(B, A) <= radii_a[None, :]).any(axis=1).mean())
    recall = float((cdist(A, B) <= radii_b[None, :]).any(axis=1).mean())
    return precision, recall


def split_report(
    embeddings: ArrayLike,
    train_indices: ArrayLike,
    test_indices: ArrayLike,
    y: ArrayLike | None = None,
    *,
    texts: list[str] | None = None,
    metric: str = "euclidean",
    n_clusters: int = 10,
    max_samples: int | None = 2000,
    random_state: int | None = 42,
) -> dict[str, float]:
    """Summarize how similar/dissimilar a train/test split is.

    Larger distribution distances (``mmd_rbf``, ``energy_distance``,
    ``wasserstein_mean``, ``sliced_wasserstein``, ``ks_mean``,
    ``frechet_distance``) and ``mean_cross_distance`` indicate a *more
    adversarial* (harder) split; values near a random split indicate a
    *balanced* split; near-zero indicates *overlap*.

    ``c2st_auc`` is a classifier two-sample test — the out-of-fold AUC of a
    logistic model trained to tell train from test points: ~0.5 means the two
    sides are indistinguishable (overlap), →1.0 means trivially separable
    (adversarial). ``manifold_precision`` / ``manifold_recall`` (k-NN support
    coverage) report how much of test lies inside the train manifold and vice
    versa.

    Alongside these train↔test *distance* metrics, ``train_diversity`` and
    ``test_diversity`` report the within-split *spread* (mean distance to each
    side's own centroid) — useful for spotting when a split concentrates one
    side (e.g. an adversarial test drawn from a single outlier cluster).

    Args:
        embeddings: array-like of shape (n_samples, n_features).
        train_indices: integer index array for the training split.
        test_indices: integer index array for the test split.
        y: array-like of labels, optional. If given, adds
            ``label_distribution_shift`` (total-variation distance between
            train/test class proportions; 0 = identical balance).
        texts: per-sample raw strings, optional and aligned with ``embeddings``.
            If given, adds ``train_text_diversity`` / ``test_text_diversity``
            (mean pairwise n-gram distance within each side), subsampled to
            ``max_samples`` for the O(n²) pairwise computation.
        metric: distance metric (default ``"euclidean"``).
        n_clusters: clusters used for the leakage statistic (default 10).
        max_samples: cap per side for the O(n²) distribution metrics
            (subsampled for speed); ``None`` uses all samples (default 2000).
        random_state: seed for subsampling (default 42).

    Returns:
        A dict of structural, geometric, distributional, diversity, and
        (optional) label metrics.
    """
    X = validate_split_inputs(embeddings, 0.5)  # reuse finite/2-D validation
    train_indices = np.asarray(train_indices, dtype=np.intp)
    test_indices = np.asarray(test_indices, dtype=np.intp)
    rng = check_random_state(random_state)

    n_train, n_test = len(train_indices), len(test_indices)
    # A report compares two non-empty sides; an empty side otherwise divides by
    # zero here or crashes downstream in compute_split_similarity.
    if n_train == 0 or n_test == 0:
        raise ValueError(
            f"split_report requires non-empty train and test splits "
            f"(got n_train={n_train}, n_test={n_test})."
        )
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
    report["sliced_wasserstein"] = _sliced_wasserstein(
        A, B, _SLICED_WASSERSTEIN_PROJECTIONS, rng
    )
    report["frechet_distance"] = _frechet_distance(A, B)

    # Learned separability (can a classifier tell train from test?) and k-NN
    # support coverage between the two manifolds.
    report["c2st_auc"] = _c2st_auc(
        A, B, random_state if isinstance(random_state, int) else 42
    )
    precision, recall = _manifold_precision_recall(A, B, _MANIFOLD_K)
    report["manifold_precision"] = precision
    report["manifold_recall"] = recall

    # Within-split diversity (spread): how varied is each side on its own.
    report["train_diversity"] = mean_dist(X[train_indices])
    report["test_diversity"] = mean_dist(X[test_indices])

    # Optional text diversity: mean pairwise n-gram distance within each side.
    if texts is not None:
        texts = list(texts)
        seed = random_state if isinstance(random_state, int) else 42
        report["train_text_diversity"] = diversity_text(
            [texts[i] for i in train_indices],
            sample_size=max_samples, random_state=seed,
        )
        report["test_text_diversity"] = diversity_text(
            [texts[i] for i in test_indices],
            sample_size=max_samples, random_state=seed,
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
