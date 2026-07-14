"""Branch-coverage tests for overlap splitters.

Targets the optimization/edge paths that the happy-path tests in
test_splitters.py don't reach: neighbor-coverage feasibility, empty
distance bins in stratified-similarity, and the greedy swap loop in
max-coverage.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from splytters.overlap import (
    max_coverage_split,
    nearest_neighbor_split,
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


class TestNeighborCoverageFeasibility:

    def test_high_feasible_k_satisfies_contract(self, two_clusters):
        """The exact feasibility path handles a demanding but achievable k."""
        train, test = neighbor_coverage_split(two_clusters, train_size=0.6, k=30)
        _disjoint_and_complete(train, test, len(two_clusters))

    def test_extreme_k_reports_infeasible_contract(self, two_clusters):
        """An impossible coverage promise must raise instead of silently lying."""
        with pytest.raises(ValueError, match="no feasible neighbor-coverage split"):
            neighbor_coverage_split(two_clusters, train_size=0.6, k=1000)

    @pytest.mark.parametrize("k", [0, "1"])
    def test_invalid_k_is_rejected(self, k):
        with pytest.raises(ValueError, match="k must be a positive integer"):
            neighbor_coverage_split(np.arange(8, dtype=float).reshape(4, 2), k=k)

    def test_infeasible_neighborhood_graph_reports_the_contract_failure(self):
        """A train set can be large enough for k yet still cover no valid split."""
        X = np.array([-0.132105, 0.1049, 0.12573, 0.640423])[:, None]
        with pytest.raises(ValueError, match="median-distance neighborhood graph"):
            neighbor_coverage_split(X, train_size=1, k=1)

    def test_rejects_an_invalid_milp_result(self, monkeypatch):
        """Do not return a split if an optimizer ever violates its constraints."""
        import scipy.optimize

        monkeypatch.setattr(
            scipy.optimize,
            "milp",
            lambda **_: SimpleNamespace(success=True, x=np.array([1.0, 1.0, 1.0, 0.0])),
        )
        X = np.arange(8, dtype=float).reshape(4, 2)
        with pytest.raises(RuntimeError, match="invalid solution"):
            neighbor_coverage_split(X, train_size=0.5, k=1)

    def test_undercovered_split_warns_best_effort(self, monkeypatch):
        """A solver returning a correctly-sized but under-covered split must warn
        (best-effort) instead of silently violating the coverage promise."""
        import scipy.optimize

        # x has the right train count (3) but leaves both test points with fewer
        # than k=3 train neighbors within the median-distance neighborhood.
        monkeypatch.setattr(
            scipy.optimize,
            "milp",
            lambda **_: SimpleNamespace(
                success=True, x=np.array([1.0, 1.0, 1.0, 0.0, 0.0])
            ),
        )
        X = np.array([0.0, 0.1, 0.2, 0.3, 50.0])[:, None]
        with pytest.warns(UserWarning, match="fewer than k=3 train neighbors"):
            train, test = neighbor_coverage_split(X, train_size=0.6, k=3)
        _disjoint_and_complete(train, test, len(X))


class TestNearestNeighborDuplicates:

    @pytest.mark.parametrize("seed", [3, 4, 9])
    def test_duplicates_do_not_self_neighbor(self, seed):
        """With exact duplicates, a point must never record itself as its own
        nearest neighbor. Before the self-mask fix the higher-indexed twin's
        recorded NN was itself, which wasted a test slot and raised a spurious
        "could only place N" warning on this easily satisfiable case."""
        X = np.zeros((3, 2))  # three exact duplicates
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning fails the test
            train, test = nearest_neighbor_split(X, train_size=0.34, random_state=seed)
        _disjoint_and_complete(train, test, len(X))
        assert len(test) == 2  # full requested test size is placed


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
        train, test = max_coverage_split(X, train_size=0.5, radius=1.0, random_state=0)
        _disjoint_and_complete(train, test, len(X))

        from scipy.spatial.distance import cdist
        d = cdist(X[test], X[train])
        assert (d.min(axis=1) <= 1.0).all()  # every test point covered
