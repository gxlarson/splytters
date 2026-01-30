"""
Sorting algorithms for adversarial dataset partitioning based on embeddings.

These functions rank samples by embedding-based criteria (distance to centroid,
nearest neighbors, density, outlier scores) to enable train-test splits that
maximize dissimilarity.
"""

from pprint import pprint

import numpy as np
from scipy.spatial.distance import cdist, euclidean as _dist_euclidean
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def dist_euclidean(u, v):
    """Compute Euclidean distance between two vectors."""
    return _dist_euclidean(u, v)


def distance_to_mean(embeddings, distance=dist_euclidean):
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


def distance_to_nearest_neighbor(embeddings, metric="euclidean"):
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

    # Compute pairwise distances
    pairwise_dist = cdist(embeddings, embeddings, metric=metric)

    # Set diagonal to infinity so we don't count self as nearest neighbor
    np.fill_diagonal(pairwise_dist, np.inf)

    # Find minimum distance for each sample
    min_distances = pairwise_dist.min(axis=1)

    # Create sorted list of (index, distance)
    scores = [(i, min_distances[i]) for i in range(len(embeddings))]
    scores.sort(key=lambda p: p[1])

    return scores


def local_density(embeddings, radius=None, metric="euclidean", low_first=True):
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

    # Compute pairwise distances
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


def outlier_score(embeddings, method="isolation_forest", low_first=True, **kwargs):
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
