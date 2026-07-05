"""Unit tests for embedding_sorters.py"""

import numpy as np
import pytest

# Lazy per-modality imports mean this pulls in only numpy/scikit-learn.
from splytters.sorters.embedding_sorters import (
    dist_euclidean,
    distance_to_mean,
    distance_to_nearest_neighbor,
    local_density,
    outlier_score,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def embeddings():
    """Simple 2D embeddings with a clear structure:
    cluster near origin + one outlier far away."""
    return np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [10.0, 10.0],  # outlier
    ])


@pytest.fixture
def embeddings_large():
    """50 points from two clusters for density/outlier tests."""
    rng = np.random.RandomState(42)
    cluster = rng.randn(45, 2) * 0.5
    outliers = rng.randn(5, 2) * 0.5 + np.array([10, 10])
    return np.vstack([cluster, outliers])


# ===========================================================================
# dist_euclidean
# ===========================================================================

class TestDistEuclidean:

    def test_identical_vectors(self):
        assert dist_euclidean([0, 0], [0, 0]) == 0.0

    def test_known_distance(self):
        assert dist_euclidean([0, 0], [3, 4]) == pytest.approx(5.0)

    def test_symmetric(self):
        u, v = [1, 2], [4, 6]
        assert dist_euclidean(u, v) == pytest.approx(dist_euclidean(v, u))


# ===========================================================================
# distance_to_mean
# ===========================================================================

class TestDistanceToMean:

    def test_returns_index_distance_tuples(self, embeddings):
        results = distance_to_mean(embeddings)
        assert len(results) == len(embeddings)
        for idx, dist in results:
            assert isinstance(idx, int)
            assert isinstance(dist, float)

    def test_sorted_ascending(self, embeddings):
        results = distance_to_mean(embeddings)
        distances = [d for _, d in results]
        assert distances == sorted(distances)

    def test_outlier_ranked_last(self, embeddings):
        """The point at (10,10) should be furthest from the centroid."""
        results = distance_to_mean(embeddings)
        assert results[-1][0] == 4

    def test_closest_to_centroid_ranked_first(self):
        """Point at the exact centroid should have distance ~0."""
        X = np.array([[0, 0], [2, 0], [0, 2], [2, 2]])
        # centroid is (1, 1)
        results = distance_to_mean(X)
        # All equidistant, so just check distances are equal
        distances = [d for _, d in results]
        assert all(d == pytest.approx(distances[0]) for d in distances)

    def test_empty(self):
        X = np.empty((0, 2))
        results = distance_to_mean(X)
        assert results == []


# ===========================================================================
# distance_to_nearest_neighbor
# ===========================================================================

class TestDistanceToNearestNeighbor:

    def test_returns_index_distance_tuples(self, embeddings):
        results = distance_to_nearest_neighbor(embeddings)
        assert len(results) == len(embeddings)
        for idx, dist in results:
            assert isinstance(idx, int)
            assert isinstance(dist, float)
            assert dist > 0

    def test_sorted_ascending(self, embeddings):
        results = distance_to_nearest_neighbor(embeddings)
        distances = [d for _, d in results]
        assert distances == sorted(distances)

    def test_outlier_has_largest_nn_distance(self, embeddings):
        """The outlier at (10,10) should have the largest NN distance."""
        results = distance_to_nearest_neighbor(embeddings)
        assert results[-1][0] == 4

    def test_close_points_ranked_first(self, embeddings):
        """Points in the cluster near origin should have small NN distances."""
        results = distance_to_nearest_neighbor(embeddings)
        cluster_indices = {0, 1, 2, 3}
        # First 4 results should all be from the cluster
        top_4 = {idx for idx, _ in results[:4]}
        assert top_4 == cluster_indices

    def test_two_points(self):
        X = np.array([[0, 0], [3, 4]])
        results = distance_to_nearest_neighbor(X)
        assert len(results) == 2
        assert results[0][1] == pytest.approx(5.0)
        assert results[1][1] == pytest.approx(5.0)


# ===========================================================================
# local_density
# ===========================================================================

class TestLocalDensity:

    def test_returns_index_count_tuples(self, embeddings):
        results = local_density(embeddings)
        assert len(results) == len(embeddings)
        for idx, count in results:
            assert isinstance(idx, int)
            assert isinstance(count, int)
            assert count >= 0

    def test_low_first_sparse_first(self, embeddings):
        """With low_first=True, sparse (low density) points come first."""
        results = local_density(embeddings, low_first=True)
        # Outlier at index 4 should be among the sparsest
        assert results[0][0] == 4

    def test_low_first_false_dense_first(self, embeddings):
        """With low_first=False, dense points come first."""
        results = local_density(embeddings, low_first=False)
        # Outlier should be last
        assert results[-1][0] == 4

    def test_custom_radius(self, embeddings):
        """Very small radius should give 0 neighbors for most points."""
        results = local_density(embeddings, radius=0.01)
        counts = {idx: count for idx, count in results}
        assert counts[4] == 0  # outlier has no neighbors within 0.01

    def test_large_radius_all_neighbors(self, embeddings):
        """Large radius should give max neighbors for all points."""
        results = local_density(embeddings, radius=100.0)
        for _, count in results:
            assert count == len(embeddings) - 1


# ===========================================================================
# outlier_score
# ===========================================================================

class TestOutlierScore:

    def test_returns_index_score_tuples(self, embeddings_large):
        results = outlier_score(embeddings_large)
        assert len(results) == len(embeddings_large)
        for idx, score in results:
            assert isinstance(idx, int)
            assert isinstance(score, float)

    def test_isolation_forest_finds_outliers(self, embeddings_large):
        """The 5 outlier points (indices 45-49) should have high scores."""
        results = outlier_score(embeddings_large, method="isolation_forest", low_first=True)
        outlier_indices = set(range(45, 50))
        # The last 5 should be the outliers
        top_outliers = {idx for idx, _ in results[-5:]}
        assert len(top_outliers & outlier_indices) >= 4  # at least 4 of 5

    def test_lof_finds_outliers(self, embeddings_large):
        """LOF should also rank outlier points high."""
        results = outlier_score(embeddings_large, method="lof", low_first=True)
        outlier_indices = set(range(45, 50))
        top_outliers = {idx for idx, _ in results[-5:]}
        assert len(top_outliers & outlier_indices) >= 4

    def test_low_first_true_inliers_first(self, embeddings_large):
        """With low_first=True, inliers (low score) come first."""
        results = outlier_score(embeddings_large, low_first=True)
        # First result should be from the main cluster (indices 0-44)
        assert results[0][0] < 45

    def test_low_first_false_outliers_first(self, embeddings_large):
        """With low_first=False, outliers (high score) come first."""
        results = outlier_score(embeddings_large, low_first=False)
        # First result should be from outlier group
        assert results[0][0] >= 45

    def test_invalid_method_raises(self, embeddings):
        with pytest.raises(ValueError, match="Unknown outlier detection method"):
            outlier_score(embeddings, method="invalid")

    def test_random_state_kwarg_does_not_collide(self, embeddings_large):
        """random_state is a named parameter now, so passing it no longer clashes
        with a hardcoded seed (previously TypeError: multiple values)."""
        results = outlier_score(
            embeddings_large, method="isolation_forest", random_state=7
        )
        assert len(results) == len(embeddings_large)

    def test_random_state_is_reproducible(self, embeddings_large):
        a = outlier_score(embeddings_large, method="isolation_forest", random_state=7)
        b = outlier_score(embeddings_large, method="isolation_forest", random_state=7)
        assert a == b


class TestNearestNeighborScalability:
    """distance_to_nearest_neighbor uses NearestNeighbors (O(n·k) memory)."""

    def test_matches_bruteforce(self):
        from scipy.spatial.distance import cdist

        rng = np.random.RandomState(0)
        X = rng.randn(60, 8)
        got = dict(distance_to_nearest_neighbor(X))

        D = cdist(X, X)
        np.fill_diagonal(D, np.inf)
        ref = D.min(axis=1)
        for i in range(60):
            assert got[i] == pytest.approx(ref[i], rel=1e-9)

    def test_sorted_ascending(self):
        rng = np.random.RandomState(2)
        vals = [d for _, d in distance_to_nearest_neighbor(rng.randn(40, 5))]
        assert vals == sorted(vals)

    def test_runs_on_large_n(self):
        # A full O(n²) matrix here would allocate ~0.5 GB; the NN path does not.
        rng = np.random.RandomState(1)
        got = distance_to_nearest_neighbor(rng.randn(8000, 12))
        assert len(got) == 8000

    def test_single_sample(self):
        got = distance_to_nearest_neighbor(np.zeros((1, 3)))
        assert got == [(0, float("inf"))]
