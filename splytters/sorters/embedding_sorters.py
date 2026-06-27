"""
Sorting algorithms for adversarial dataset partitioning based on embeddings.

These functions rank samples by embedding-based criteria (distance to centroid,
nearest neighbors, density, outlier scores) to enable train-test splits that
maximize dissimilarity.
"""

from __future__ import annotations

from collections.abc import Callable
from pprint import pprint
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean as _dist_euclidean
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def dist_euclidean(u: ArrayLike, v: ArrayLike) -> float:
    """Compute Euclidean distance between two vectors."""
    return _dist_euclidean(u, v)


def distance_to_mean(
    embeddings: np.ndarray,
    distance: Callable[[np.ndarray, np.ndarray], float] = dist_euclidean,
) -> list[tuple[int, float]]:
    """
    Sort samples by distance from the dataset centroid.

    Samples closest to the centroid (most "typical") appear first.
    Useful for adversarial splits: assign nearby samples to train,
    distant samples to test.

    Args:
        embeddings: np.array of shape (n_samples, embedding_dim)
        distance: distance function taking two vectors, default Euclidean

    Returns:
        List of (index, distance) tuples sorted by distance ascending.
    """
    (n, d) = embeddings.shape
    centroid = embeddings.mean(0)
    distances = []
    for i in range(n):
        dist = distance(embeddings[i], centroid)
        distances.append((i, dist))
    distances.sort(key=lambda p: p[1])
    return distances


def distance_to_nearest_neighbor(
    embeddings: ArrayLike, metric: str = "euclidean"
) -> list[tuple[int, float]]:
    """
    Sort samples by distance to their nearest neighbor.

    Samples in dense regions (close to neighbors) appear first.
    Isolated samples (far from all neighbors) appear last.

    Useful for adversarial splits: train on samples in dense clusters,
    test on isolated/unique samples.

    Args:
        embeddings: np.array of shape (n_samples, embedding_dim)
        metric: distance metric for cdist (default 'euclidean')

    Returns:
        List of (index, distance) tuples sorted by nearest neighbor distance ascending.
    """
    embeddings = np.asarray(embeddings)
    n = len(embeddings)
    if n < 2:
        # No neighbor to measure against.
        return [(i, float("inf")) for i in range(n)]

    # O(n·k) memory via NearestNeighbors instead of a full O(n²) matrix.
    # Querying the fit data puts each point's self at column 0 (distance 0),
    # so column 1 is exactly the 1-nearest-neighbor distance.
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=2, metric=metric)
    nn.fit(embeddings)
    dists, _ = nn.kneighbors(embeddings)
    min_distances = dists[:, 1]

    scores = [(i, float(min_distances[i])) for i in range(n)]
    scores.sort(key=lambda p: p[1])

    return scores


def local_density(
    embeddings: ArrayLike,
    radius: float | None = None,
    metric: str = "euclidean",
    low_first: bool = True,
) -> list[tuple[int, int]]:
    """
    Sort samples by local density (number of neighbors within radius).

    Samples in dense regions have many neighbors; isolated samples have few.

    Useful for adversarial splits: train on high-density regions,
    test on sparse/low-density regions.

    Args:
        embeddings: np.array of shape (n_samples, embedding_dim)
        radius: distance threshold for counting neighbors.
                If None, uses median pairwise distance.
        metric: distance metric for cdist (default 'euclidean')
        low_first: if True, sparse/isolated samples first;
                   if False, dense samples first

    Returns:
        List of (index, neighbor_count) tuples sorted by density.
    """
    embeddings = np.asarray(embeddings)

    # TODO: Replace full pairwise matrix with BallTree.query_radius to count
    # neighbors without materializing O(n²) distances.
    pairwise_dist = cdist(embeddings, embeddings, metric=metric)

    # Set diagonal to infinity so we don't count self
    np.fill_diagonal(pairwise_dist, np.inf)

    # Auto-select radius if not provided
    if radius is None:
        # Use median of all pairwise distances as default radius
        finite_dists = pairwise_dist[pairwise_dist < np.inf]
        radius = np.median(finite_dists)

    # Count neighbors within radius for each sample
    neighbor_counts = (pairwise_dist <= radius).sum(axis=1)

    # Create sorted list of (index, count)
    scores = [(i, int(neighbor_counts[i])) for i in range(len(embeddings))]
    scores.sort(key=lambda p: p[1], reverse=not low_first)

    return scores


def outlier_score(
    embeddings: ArrayLike,
    method: str = "isolation_forest",
    low_first: bool = True,
    **kwargs: Any,
) -> list[tuple[int, float]]:
    """
    Sort samples by anomaly/outlier score.

    Higher outlier scores indicate samples that are unusual or don't fit
    the overall data distribution.

    Useful for adversarial splits: train on normal/typical samples,
    test on outliers/anomalies.

    Args:
        embeddings: np.array of shape (n_samples, embedding_dim)
        method: outlier detection algorithm, one of:
            - 'isolation_forest': Isolation Forest (fast, good for high dimensions)
            - 'lof': Local Outlier Factor (density-based)
        low_first: if True, normal/inlier samples first;
                   if False, outliers/anomalies first
        **kwargs: additional arguments passed to the outlier detector

    Returns:
        List of (index, outlier_score) tuples sorted by outlier score.
        For isolation_forest: more negative = more normal, more positive = more outlier.
        For lof: scores > 1 indicate outliers, < 1 indicate inliers.
    """
    embeddings = np.asarray(embeddings)

    if method == "isolation_forest":
        detector = IsolationForest(random_state=42, **kwargs)
        detector.fit(embeddings)
        # score_samples returns negative scores; more negative = more normal
        # We negate so higher = more outlier
        raw_scores = -detector.score_samples(embeddings)

    elif method == "lof":
        detector = LocalOutlierFactor(novelty=False, **kwargs)
        detector.fit_predict(embeddings)
        # negative_outlier_factor_ is negative; more negative = more outlier
        # We negate so higher = more outlier
        raw_scores = -detector.negative_outlier_factor_

    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

    # Create sorted list of (index, score)
    scores = [(i, raw_scores[i]) for i in range(len(embeddings))]
    scores.sort(key=lambda p: p[1], reverse=not low_first)

    return scores


def mahalanobis_distance_to_mean(
    embeddings: ArrayLike, low_first: bool = True
) -> list[tuple[int, float]]:
    """
    Sort samples by Mahalanobis distance from the dataset centroid.

    Like :func:`distance_to_mean`, but covariance-aware: atypicality is measured
    relative to the data's own spread and feature correlations rather than in
    raw Euclidean units. Most typical samples first.

    Args:
        embeddings: array of shape (n_samples, embedding_dim).
        low_first: if True, the most typical (smallest distance) samples first.

    Returns:
        List of (index, mahalanobis_distance) tuples.
    """
    X = np.asarray(embeddings, dtype=float)
    centroid = X.mean(axis=0)
    cov = np.atleast_2d(np.cov(X, rowvar=False)) + 1e-6 * np.eye(X.shape[1])
    inv_cov = np.linalg.pinv(cov)
    diff = X - centroid
    d2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    dists = np.sqrt(np.maximum(d2, 0.0))
    scores = sorted(
        enumerate(dists.tolist()), key=lambda p: p[1], reverse=not low_first
    )
    return [(i, float(v)) for i, v in scores]


def knn_label_disagreement(
    embeddings: ArrayLike,
    y: ArrayLike,
    k: int = 5,
    metric: str = "euclidean",
    low_first: bool = True,
) -> list[tuple[int, float]]:
    """
    Sort samples by the fraction of their k nearest neighbors with a *different*
    label — a class-boundary / ambiguity score.

    A point surrounded by same-label neighbors is class-typical (interior); one
    whose neighbors disagree sits on a boundary and is harder. This is the
    label-aware (supervised) sorter — pair it with
    :func:`splytters.sorted_stratified_split` for a difficulty curriculum.

    Args:
        embeddings: array of shape (n_samples, embedding_dim).
        y: class labels of shape (n_samples,).
        k: number of neighbors to inspect (the point itself is excluded).
        metric: distance metric for the neighbor search.
        low_first: if True, the most class-typical (low-disagreement) first.

    Returns:
        List of (index, disagreement_fraction) tuples, each in [0, 1].
    """
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(embeddings)
    y = np.asarray(y)
    n = len(X)
    k = min(k, n - 1)
    neighbors = (
        NearestNeighbors(n_neighbors=k + 1, metric=metric)
        .fit(X)
        .kneighbors(X, return_distance=False)[:, 1:]  # drop self
    )
    scores = [(i, float(np.mean(y[neighbors[i]] != y[i]))) for i in range(n)]
    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


if __name__ == "__main__":
    # Example: sort texts by distance to mean embedding.
    # Texts farther from the centroid (more atypical) appear later.
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    texts = [
        "what is my balance",
        "my balance is what",
        "how much do I owe",
        "what's my balance"
    ]
    embeddings = embedder.encode(texts)
    distances = distance_to_mean(embeddings)
    distances = [(texts[i], d) for (i, d) in distances]
    pprint(distances)
