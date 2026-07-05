"""Branch-coverage tests for overlap splitters.

Targets the optimization/edge paths that the happy-path tests in
test_splitters.py don't reach: the neighbor-coverage swap logic, empty
distance bins in stratified-similarity, and the greedy swap loop in
max-coverage.
"""

import numpy as np
import pytest

from splytters.overlap import (
    max_coverage_split,
    neighbor_coverage_split,
    stratified_similarity_split,
)


def _disjoint_and_complete(train, test, n):
    assert isinstance(train, np.ndarray) and isinstance(test, np.ndarray)
    assert set(train.tolist()) & set(test.tolist()) == set()
    assert set(train.tolist()) | set(test.tolist()) == set(range(n))


@pytest.fixture
def two_clusters():
    """120 points in two tight, well-separated 2D clusters."""
    rng = np.random.RandomState(42)
    a = rng.randn(60, 2) * 0.3 + np.array([6, 6])
    b = rng.randn(60, 2) * 0.3 + np.array([-6, -6])
    return np.vstack([a, b])


class TestNeighborCoverageSwap:

    def test_high_k_triggers_swap_logic(self, two_clusters):
        """A large k forces the 'not enough similar in train' swap branch."""
        train, test = neighbor_coverage_split(two_clusters, train_size=0.6, k=40)
        _disjoint_and_complete(train, test, len(two_clusters))

    def test_extreme_k_no_viable_swap(self, two_clusters):
        """k above any achievable redundancy exercises the no-swap fallback."""
        train, test = neighbor_coverage_split(two_clusters, train_size=0.6, k=1000)
        _disjoint_and_complete(train, test, len(two_clusters))


class TestStratifiedSimilarityEmptyBins:

    def test_more_bins_than_points_leaves_empty_bins(self):
        """n_bins far exceeding sample count guarantees empty bins (the
        `continue` branch)."""
        rng = np.random.RandomState(0)
        X = rng.randn(12, 2)
        train, test = stratified_similarity_split(X, train_size=0.7, n_bins=50)
        _disjoint_and_complete(train, test, len(X))


class TestMaxCoverageSwap:

    def test_small_radius_runs_greedy_swaps(self):
        """A radius covering only within tight pairs leaves some test points
        uncovered, driving the greedy swap search."""
        # Three tight pairs, far apart: within-pair distance ~0.1, between >10.
        X = np.array([
            [0.0, 0.0], [0.1, 0.0],
            [10.0, 10.0], [10.1, 10.0],
            [20.0, 20.0], [20.1, 20.0],
            [30.0, 30.0], [30.1, 30.0],
        ])
        train, test = max_coverage_split(X, train_size=0.5, radius=1.0)
        _disjoint_and_complete(train, test, len(X))

    def test_tiny_radius_no_improving_swap(self):
        """A radius so small nothing covers anything: swap search finds no
        improvement (best_swap stays None)."""
        rng = np.random.RandomState(3)
        X = rng.randn(10, 2) * 10
        train, test = max_coverage_split(X, train_size=0.5, radius=1e-6)
        _disjoint_and_complete(train, test, len(X))

    def test_covers_all_test_points_when_possible(self):
        """Four tight pairs, one point per pair to each side: the greedy swaps
        should reach full coverage (every test point has a train point within
        radius). Guards the incremental cover_count rewrite for correctness."""
        X = np.array([
            [0.0, 0.0], [0.1, 0.0],
            [10.0, 10.0], [10.1, 10.0],
            [20.0, 20.0], [20.1, 20.0],
            [30.0, 30.0], [30.1, 30.0],
        ])
        train, test = max_coverage_split(X, train_size=0.5, radius=1.0)
        _disjoint_and_complete(train, test, len(X))

        from scipy.spatial.distance import cdist
        d = cdist(X[test], X[train])
        assert (d.min(axis=1) <= 1.0).all()  # every test point covered
