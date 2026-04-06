"""Unit tests for splitting algorithms."""

import numpy as np
import pytest

from splitters.utils import (
    compute_pairwise_distances,
    compute_centroid,
    compute_split_centroids,
    cluster_embeddings,
    random_split,
    compute_split_similarity,
    greedy_assign_to_target,
)
from splitters.adversarial import (
    cluster_split,
    centroid_adversarial_split,
    distance_adversarial_split,
    density_adversarial_split,
    outlier_adversarial_split,
    min_cut_split,
    normalized_cut_split,
    get_cluster_info,
)
from splitters.balanced import (
    distribution_matched_split,
    moment_matched_split,
    histogram_matched_split,
    stratified_random_split,
    density_balanced_split,
    mmd_minimized_split,
)
from splitters.overlap import (
    cluster_leak_split,
    neighbor_coverage_split,
    centroid_matched_split,
    stratified_similarity_split,
    nearest_neighbor_split,
    duplicate_spread_split,
    max_coverage_split,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def embeddings_2d():
    """100 points from two well-separated 2D clusters."""
    rng = np.random.RandomState(42)
    cluster_a = rng.randn(50, 2) + np.array([5, 5])
    cluster_b = rng.randn(50, 2) + np.array([-5, -5])
    return np.vstack([cluster_a, cluster_b])


@pytest.fixture
def embeddings_small():
    """10 simple 2D points for quick tests."""
    rng = np.random.RandomState(0)
    return rng.randn(10, 2)


@pytest.fixture
def embeddings_high_dim():
    """50 points in 64 dimensions."""
    rng = np.random.RandomState(7)
    return rng.randn(50, 64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_valid_split(train, test, n_samples, train_ratio=0.7, ratio_tol=0.15):
    """Check structural invariants that every split must satisfy."""
    # Both are lists
    assert isinstance(train, list)
    assert isinstance(test, list)

    # No duplicates within each set
    assert len(set(train)) == len(train), "Duplicate indices in train"
    assert len(set(test)) == len(test), "Duplicate indices in test"

    # Disjoint
    overlap = set(train) & set(test)
    assert len(overlap) == 0, f"Train/test overlap: {overlap}"

    # Union covers all indices
    assert set(train) | set(test) == set(range(n_samples))

    # Ratio within tolerance
    actual_ratio = len(train) / n_samples
    assert abs(actual_ratio - train_ratio) < ratio_tol, (
        f"Train ratio {actual_ratio:.2f} outside tolerance of {train_ratio} ± {ratio_tol}"
    )


# ===========================================================================
# Utils
# ===========================================================================

class TestComputePairwiseDistances:

    def test_shape(self, embeddings_small):
        D = compute_pairwise_distances(embeddings_small)
        n = len(embeddings_small)
        assert D.shape == (n, n)

    def test_symmetric(self, embeddings_small):
        D = compute_pairwise_distances(embeddings_small)
        np.testing.assert_allclose(D, D.T)

    def test_zero_diagonal(self, embeddings_small):
        D = compute_pairwise_distances(embeddings_small)
        np.testing.assert_allclose(np.diag(D), 0, atol=1e-10)


class TestComputeCentroid:

    def test_shape(self, embeddings_small):
        c = compute_centroid(embeddings_small)
        assert c.shape == (embeddings_small.shape[1],)

    def test_value(self):
        X = np.array([[0, 0], [2, 2]])
        np.testing.assert_array_equal(compute_centroid(X), [1, 1])


class TestComputeSplitCentroids:

    def test_returns_centroids(self, embeddings_small):
        train_c, test_c = compute_split_centroids(
            embeddings_small, [0, 1, 2], [3, 4, 5, 6, 7, 8, 9]
        )
        assert train_c.shape == (2,)
        assert test_c.shape == (2,)

    def test_empty_returns_none(self, embeddings_small):
        train_c, test_c = compute_split_centroids(embeddings_small, [], [0, 1])
        assert train_c is None


class TestClusterEmbeddings:

    def test_returns_correct_structure(self, embeddings_2d):
        labels, cluster_map, centers = cluster_embeddings(embeddings_2d, n_clusters=2)
        assert len(labels) == len(embeddings_2d)
        assert len(centers) == 2
        assert sum(len(v) for v in cluster_map.values()) == len(embeddings_2d)


class TestRandomSplit:

    def test_valid_split(self, embeddings_2d):
        train, test = random_split(embeddings_2d, train_ratio=0.7)
        assert_valid_split(train, test, len(embeddings_2d))

    def test_deterministic(self, embeddings_2d):
        t1, _ = random_split(embeddings_2d, random_state=0)
        t2, _ = random_split(embeddings_2d, random_state=0)
        assert t1 == t2


class TestComputeSplitSimilarity:

    def test_returns_expected_keys(self, embeddings_small):
        train, test = random_split(embeddings_small)
        result = compute_split_similarity(embeddings_small, train, test)
        assert "centroid_distance" in result
        assert "mean_cross_distance" in result
        assert "coverage" in result


class TestGreedyAssignToTarget:

    def test_basic(self):
        items = [(0, 3), (1, 4), (2, 2)]
        selected, remaining = greedy_assign_to_target(items, 5)
        assert 0 in selected  # size 3, fits
        assert 2 in selected  # size 2, total 5 fits
        assert 1 in remaining  # size 4, would exceed


# ===========================================================================
# Adversarial splitters
# ===========================================================================

class TestClusterSplit:

    def test_valid_split(self, embeddings_2d):
        train, test = cluster_split(embeddings_2d, n_clusters=5)
        assert_valid_split(train, test, len(embeddings_2d))

    def test_dbscan(self, embeddings_2d):
        train, test = cluster_split(embeddings_2d, method="dbscan", eps=2.0, min_samples=3)
        assert_valid_split(train, test, len(embeddings_2d), ratio_tol=0.3)

    def test_invalid_method(self, embeddings_small):
        with pytest.raises(ValueError, match="Unknown clustering method"):
            cluster_split(embeddings_small, method="invalid")


class TestCentroidAdversarialSplit:

    def test_valid_split(self, embeddings_2d):
        train, test = centroid_adversarial_split(embeddings_2d, n_clusters=5)
        assert_valid_split(train, test, len(embeddings_2d))

    def test_test_further_from_centroid(self, embeddings_2d):
        """Test samples should on average be further from centroid than train."""
        train, test = centroid_adversarial_split(embeddings_2d, n_clusters=5)
        centroid = embeddings_2d.mean(axis=0)
        train_dists = np.linalg.norm(embeddings_2d[train] - centroid, axis=1).mean()
        test_dists = np.linalg.norm(embeddings_2d[test] - centroid, axis=1).mean()
        assert test_dists >= train_dists


class TestDistanceAdversarialSplit:

    def test_valid_split(self, embeddings_2d):
        train, test = distance_adversarial_split(embeddings_2d)
        assert_valid_split(train, test, len(embeddings_2d))

    def test_test_further_from_centroid(self, embeddings_2d):
        train, test = distance_adversarial_split(embeddings_2d)
        centroid = embeddings_2d.mean(axis=0)
        train_max = np.linalg.norm(embeddings_2d[train] - centroid, axis=1).max()
        test_min = np.linalg.norm(embeddings_2d[test] - centroid, axis=1).min()
        # Every test sample should be at least as far as the closest train sample
        assert test_min >= train_max - 1e-10


class TestDensityAdversarialSplit:

    def test_valid_split(self, embeddings_2d):
        train, test = density_adversarial_split(embeddings_2d)
        assert_valid_split(train, test, len(embeddings_2d))


class TestOutlierAdversarialSplit:

    def test_valid_split(self, embeddings_2d):
        train, test = outlier_adversarial_split(embeddings_2d)
        assert_valid_split(train, test, len(embeddings_2d))


class TestMinCutSplit:

    def test_valid_split_spectral(self, embeddings_small):
        train, test = min_cut_split(embeddings_small, method="spectral")
        assert_valid_split(train, test, len(embeddings_small), ratio_tol=0.3)

    def test_tiny_dataset(self):
        X = np.array([[0, 0], [1, 1]])
        train, test = min_cut_split(X, train_ratio=0.5)
        assert_valid_split(train, test, 2, train_ratio=0.5, ratio_tol=0.5)

    def test_invalid_method(self, embeddings_small):
        with pytest.raises(ValueError, match="Unknown method"):
            min_cut_split(embeddings_small, method="invalid")


class TestNormalizedCutSplit:

    def test_valid_split(self, embeddings_small):
        train, test = normalized_cut_split(embeddings_small)
        assert_valid_split(train, test, len(embeddings_small), ratio_tol=0.3)

    def test_tiny_dataset(self):
        X = np.array([[0, 0], [1, 1]])
        train, test = normalized_cut_split(X, train_ratio=0.5)
        assert_valid_split(train, test, 2, train_ratio=0.5, ratio_tol=0.5)


class TestGetClusterInfo:

    def test_returns_expected_keys(self, embeddings_2d):
        train, test = random_split(embeddings_2d)
        info = get_cluster_info(embeddings_2d, train, test, n_clusters=5)
        assert "cluster_stats" in info
        assert "clusters_with_leakage" in info
        assert "leakage_ratio" in info
        assert len(info["cluster_stats"]) == 5


# ===========================================================================
# Balanced splitters
# ===========================================================================

class TestDistributionMatchedSplit:

    def test_valid_split(self, embeddings_2d):
        train, test = distribution_matched_split(
            embeddings_2d, n_iterations=50
        )
        assert_valid_split(train, test, len(embeddings_2d))


class TestMomentMatchedSplit:

    def test_valid_split(self, embeddings_2d):
        train, test = moment_matched_split(embeddings_2d, n_iterations=50)
        assert_valid_split(train, test, len(embeddings_2d))

    def test_centroids_close(self, embeddings_2d):
        """After optimization, train/test centroids should be close."""
        train, test = moment_matched_split(
            embeddings_2d, n_iterations=200
        )
        train_mean = embeddings_2d[train].mean(axis=0)
        test_mean = embeddings_2d[test].mean(axis=0)
        dist = np.linalg.norm(train_mean - test_mean)
        # Should be closer than a random split's centroids (loose check)
        assert dist < 5.0


class TestHistogramMatchedSplit:

    def test_valid_split(self, embeddings_2d):
        train, test = histogram_matched_split(
            embeddings_2d, n_iterations=50
        )
        assert_valid_split(train, test, len(embeddings_2d))


class TestStratifiedRandomSplit:

    def test_valid_split(self, embeddings_2d):
        labels = np.array([0] * 50 + [1] * 50)
        train, test = stratified_random_split(embeddings_2d, labels)
        assert_valid_split(train, test, len(embeddings_2d))

    def test_preserves_proportions(self, embeddings_2d):
        labels = np.array([0] * 50 + [1] * 50)
        train, test = stratified_random_split(embeddings_2d, labels, train_ratio=0.8)
        train_labels = labels[train]
        # Each class should have ~80% in train
        for cls in [0, 1]:
            cls_in_train = (train_labels == cls).sum()
            assert abs(cls_in_train / 50 - 0.8) < 0.05


class TestDensityBalancedSplit:

    def test_valid_split(self, embeddings_2d):
        train, test = density_balanced_split(embeddings_2d)
        assert_valid_split(train, test, len(embeddings_2d), ratio_tol=0.2)


class TestMmdMinimizedSplit:

    def test_valid_split(self, embeddings_2d):
        train, test = mmd_minimized_split(embeddings_2d, n_iterations=50)
        assert_valid_split(train, test, len(embeddings_2d))


# ===========================================================================
# Overlap splitters
# ===========================================================================

class TestClusterLeakSplit:

    def test_valid_split(self, embeddings_2d):
        train, test = cluster_leak_split(embeddings_2d, n_clusters=5)
        assert_valid_split(train, test, len(embeddings_2d), ratio_tol=0.2)

    def test_clusters_leak(self, embeddings_2d):
        """Each cluster should have members in both train and test."""
        train, test = cluster_leak_split(embeddings_2d, n_clusters=5)
        info = get_cluster_info(embeddings_2d, train, test, n_clusters=5)
        # Most clusters should leak by design
        assert info["leakage_ratio"] > 0.5


class TestNeighborCoverageSplit:

    def test_valid_split(self, embeddings_small):
        train, test = neighbor_coverage_split(embeddings_small, k=2)
        assert_valid_split(train, test, len(embeddings_small), ratio_tol=0.3)


class TestCentroidMatchedSplit:

    def test_valid_split(self, embeddings_2d):
        train, test = centroid_matched_split(embeddings_2d, n_iterations=50)
        assert_valid_split(train, test, len(embeddings_2d))


class TestStratifiedSimilaritySplit:

    def test_valid_split(self, embeddings_2d):
        train, test = stratified_similarity_split(embeddings_2d)
        assert_valid_split(train, test, len(embeddings_2d), ratio_tol=0.2)


class TestNearestNeighborSplit:

    def test_valid_split(self, embeddings_2d):
        train, test = nearest_neighbor_split(embeddings_2d)
        assert_valid_split(train, test, len(embeddings_2d), ratio_tol=0.2)

    def test_most_nn_in_train(self, embeddings_2d):
        """Most test samples should have their nearest neighbor in train."""
        train, test = nearest_neighbor_split(embeddings_2d)
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=2, algorithm="auto")
        nn.fit(embeddings_2d)
        neighbors = nn.kneighbors(embeddings_2d, return_distance=False)[:, 1]

        train_set = set(train)
        nn_in_train = sum(1 for t_idx in test if neighbors[t_idx] in train_set)
        # The greedy algorithm should achieve high coverage, though not 100%
        assert nn_in_train / len(test) > 0.8


class TestDuplicateSpreadSplit:

    def test_valid_split(self, embeddings_2d):
        train, test = duplicate_spread_split(embeddings_2d)
        assert_valid_split(train, test, len(embeddings_2d), ratio_tol=0.3)


class TestMaxCoverageSplit:

    def test_valid_split(self, embeddings_small):
        train, test = max_coverage_split(embeddings_small)
        assert_valid_split(train, test, len(embeddings_small), ratio_tol=0.3)


# ===========================================================================
# Cross-cutting: determinism & list input
# ===========================================================================

class TestDeterminism:
    """All splitters with random_state should be deterministic."""

    def test_random_split(self, embeddings_2d):
        a = random_split(embeddings_2d, random_state=1)
        b = random_split(embeddings_2d, random_state=1)
        assert a == b

    def test_cluster_split(self, embeddings_2d):
        a = cluster_split(embeddings_2d, random_state=1)
        b = cluster_split(embeddings_2d, random_state=1)
        assert a == b

    def test_distribution_matched(self, embeddings_2d):
        a = distribution_matched_split(embeddings_2d, n_iterations=20, random_state=1)
        b = distribution_matched_split(embeddings_2d, n_iterations=20, random_state=1)
        assert a == b

    def test_nearest_neighbor(self, embeddings_2d):
        a = nearest_neighbor_split(embeddings_2d, random_state=1)
        b = nearest_neighbor_split(embeddings_2d, random_state=1)
        assert a == b


class TestListInput:
    """Splitters should accept plain Python lists, not just numpy arrays."""

    def test_random_split_with_list(self):
        X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        train, test = random_split(X, train_ratio=0.6)
        assert_valid_split(train, test, 5, train_ratio=0.6, ratio_tol=0.3)

    def test_cluster_split_with_list(self):
        X = [[i, i] for i in range(20)]
        train, test = cluster_split(X, n_clusters=2)
        assert_valid_split(train, test, 20, ratio_tol=0.3)
