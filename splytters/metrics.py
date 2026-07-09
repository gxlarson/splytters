"""
Diversity metrics for split comparison.

These quantify the *spread* of a set of samples — how varied train or test is
on its own — complementing the train/test *distance* metrics in
:mod:`splytters.report`. Used to compare split strategies (e.g. random vs
adversarial) on coverage/diversity, not just separation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from splytters.distances import ngram_jaccard_distance


def mean_dist(
    embeddings: np.ndarray,
    distance: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> float:
    """Mean distance from each sample to the centroid — a spread/diversity score.

    With the default (Euclidean) distance this is the mean L2 distance of points
    to their centroid: larger means the set is more spread out. Pass a custom
    symmetric ``distance(u, v)`` to override; the default path is vectorized.

    Args:
        embeddings: array of shape (n_samples, n_features).
        distance: optional callable ``d(u, v) -> float``. If ``None`` (default),
            uses a vectorized Euclidean distance.

    Returns:
        Mean distance to the centroid (``0.0`` for an empty set).
    """
    X = np.asarray(embeddings, dtype=float)
    if X.ndim != 2:
        raise ValueError("embeddings must be 2-D (n_samples, n_features)")
    if len(X) == 0:
        return 0.0
    centroid = X.mean(axis=0)
    if distance is None:
        return float(np.linalg.norm(X - centroid, axis=1).mean())
    return float(np.mean([distance(centroid, row) for row in X]))


def diversity_text(
        data: list[str],
        distance_function: Callable[[str, str], float] = ngram_jaccard_distance,
        sample_size: int | None = None,
        random_state: int = 42,
    ) -> float:
    """Mean pairwise distance over a text corpus (a diversity score).

    See Figure 5 from Larson et al. (2019),
    https://aclanthology.org/N19-1051.pdf — ``D(*,*)`` is ``distance_function``
    applied to every pair of samples.

    Args:
        data: list of strings.
        distance_function: symmetric string distance, ``d(a, b) == d(b, a)``.
        sample_size: if given and smaller than ``len(data)``, estimate the mean
            on a random subsample to avoid the O(n²) full pairwise computation.
        random_state: seed for the optional subsample.

    Returns:
        Mean distance over unordered pairs (``d(a, a) == 0`` excluded).
    """
    X = list(data)
    n = len(X)
    if n < 2:
        return 0.0

    if sample_size is not None and sample_size < n:
        rng = np.random.RandomState(random_state)
        X = [X[i] for i in rng.choice(n, size=sample_size, replace=False)]
        n = sample_size

    # Symmetric distance: sum the upper triangle once, then average over the
    # n*(n-1)/2 unordered pairs (half the work of the full double loop).
    tally = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            tally += distance_function(X[i], X[j])
    return tally / (n * (n - 1) / 2)
