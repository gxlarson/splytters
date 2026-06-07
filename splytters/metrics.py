from __future__ import annotations

from collections.abc import Callable
from statistics import mean

import numpy as np

from splytters.distances import (
    dist_euclidean,
    ngram_jaccard_distance,
)


def simple_tokenizer(s: str) -> list[str]:
    return s.split()

def mean_dist(
    embeddings: np.ndarray,
    distance: Callable[[np.ndarray, np.ndarray], float] = dist_euclidean,
) -> float:
    """
    computes mean distance from all samples to the centroid

    this is sample variance when euclidean distance is used
    """
    (n, d) = embeddings.shape
    centroid = embeddings.mean(0)
    distances = []
    for i in range(n):
        distances.append(distance(centroid, embeddings[i]))
    return mean(distances)

def diversity_text(
        data: list[str],
        distance_function: Callable[[str, str], float] = ngram_jaccard_distance,
        sample_size: int | None = None,
        random_state: int = 42,
    ) -> float:
    """Mean pairwise distance over a text corpus (a diversity score).

    See Figure 5 from https://aclanthology.org/N19-1051.pdf — ``D(*,*)`` is
    ``distance_function`` applied to every pair of samples.

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

if __name__ == "__main__":
    texts = ["how much money do i have", "my balance is what", "balance is my what"]
    d = diversity_text(texts)
    print(d)
