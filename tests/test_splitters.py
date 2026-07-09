"""Unit tests for splitting algorithms."""

import warnings

import numpy as np
import pytest

from splytters.adversarial import (
    centroid_adversarial_split,
    class_boundary_split,
    cluster_kfold,
    cluster_split,
    decision_boundary_split,
    density_adversarial_split,
    distance_adversarial_split,
    get_cluster_info,
    maximin_split,
    min_cut_split,
    minority_grow_split,
    minority_split,
    mmd_maximized_split,
    normalized_cut_split,
    outlier_adversarial_split,
    wasserstein_adversarial_split,
)
from splytters.balanced import (
    density_balanced_split,
    distribution_matched_split,
    histogram_matched_split,
    mmd_minimized_split,
    moment_matched_split,
    stratified_random_split,
)
from splytters.overlap import (
    centroid_matched_split,
    cluster_leak_split,
    duplicate_spread_split,
    max_coverage_split,
    nearest_neighbor_split,
    neighbor_coverage_split,
    stratified_similarity_split,
)
from splytters.utils import (
    apportion_train,
    cluster_embeddings,
    compute_centroid,
    compute_pairwise_distances,
    compute_split_centroids,
    compute_split_similarity,
    greedy_assign_to_target,
    random_split,
    resolve_n_train,
    validate_split_inputs,
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

def assert_valid_split(train, test, n_samples, train_size=0.7, ratio_tol=0.15):
    """Check structural invariants that every split must satisfy."""
    # Both are ndarrays of indices (sklearn-style return)
    assert isinstance(train, np.ndarray)
    assert isinstance(test, np.ndarray)

    # No duplicates within each set
    assert len(set(train.tolist())) == len(train), "Duplicate indices in train"
    assert len(set(test.tolist())) == len(test), "Duplicate indices in test"

    # Disjoint
    overlap = set(train.tolist()) & set(test.tolist())
    assert len(overlap) == 0, f"Train/test overlap: {overlap}"

    # Union covers all indices
    assert set(train.tolist()) | set(test.tolist()) == set(range(n_samples))

    # Ratio within tolerance
    actual_ratio = len(train) / n_samples
    assert abs(actual_ratio - train_size) < ratio_tol, (
        f"Train ratio {actual_ratio:.2f} outside tolerance of {train_size} ± {ratio_tol}"
    )


def _splits_equal(a, b):
    """Compare two (train, test) splits of ndarrays for exact equality."""
    return np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


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
        train, test = random_split(embeddings_2d, train_size=0.7)
        assert_valid_split(train, test, len(embeddings_2d))

    def test_deterministic(self, embeddings_2d):
        t1, _ = random_split(embeddings_2d, random_state=0)
        t2, _ = random_split(embeddings_2d, random_state=0)
        assert np.array_equal(t1, t2)


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


class TestClusterSplitStrategies:
    """The strategy= assignment policies on cluster_split."""

    @pytest.fixture
    def labels(self):
        # Two classes, evenly interleaved across the 100 points.
        return np.array([0, 1] * 50)

    def test_unknown_strategy_raises(self, embeddings_2d):
        with pytest.raises(ValueError, match="Unknown strategy"):
            cluster_split(embeddings_2d, strategy="bogus")

    def test_centroid_strategy_matches_wrapper(self, embeddings_2d):
        """strategy='centroid' must equal the centroid_adversarial_split wrapper."""
        a_tr, a_te = cluster_split(embeddings_2d, n_clusters=5, strategy="centroid")
        b_tr, b_te = centroid_adversarial_split(embeddings_2d, n_clusters=5)
        assert np.array_equal(a_tr, b_tr)
        assert np.array_equal(a_te, b_te)

    def test_closest_strategy_valid(self, embeddings_2d):
        train, test = cluster_split(embeddings_2d, n_clusters=8, strategy="closest")
        assert_valid_split(train, test, len(embeddings_2d), ratio_tol=0.25)

    def test_subset_sum_requires_y(self, embeddings_2d):
        with pytest.raises(ValueError, match="requires class labels"):
            cluster_split(embeddings_2d, strategy="subset_sum")

    def test_subset_sum_y_length_mismatch(self, embeddings_2d):
        with pytest.raises(ValueError, match="length"):
            cluster_split(
                embeddings_2d, strategy="subset_sum", y=np.array([0, 1, 0])
            )

    def test_subset_sum_valid_and_class_balanced(self, embeddings_2d, labels):
        train, test = cluster_split(
            embeddings_2d, n_clusters=8, strategy="subset_sum", y=labels
        )
        assert_valid_split(train, test, len(embeddings_2d), ratio_tol=0.4)
        # Test-set class proportions should track the global proportions.
        global_p = labels.mean()
        test_p = labels[test].mean()
        assert abs(test_p - global_p) < 0.2


class TestClusterSplitFaithfulZuefle:
    """Paper-faithful opt-in modes: cluster_range k-search and individual fill.

    Züfle, Dankers & Titov (2023), "Latent Feature-based Data Splits to Improve
    Generalisation Evaluation" (GenBench @ EMNLP). The defaults must reproduce
    the pre-existing lightweight behavior exactly.
    """

    @pytest.fixture
    def labels(self):
        return np.array([0, 1] * 50)

    def test_default_matches_fixed_k_unchanged(self, embeddings_2d, labels):
        """cluster_range=None / fill_individual=False leaves the fixed-k path intact."""
        base_tr, base_te = cluster_split(
            embeddings_2d, n_clusters=8, strategy="closest", random_state=42
        )
        # Passing the new opt-out defaults explicitly must not change anything.
        same_tr, same_te = cluster_split(
            embeddings_2d, n_clusters=8, strategy="closest", random_state=42,
            cluster_range=None, fill_individual=False,
        )
        assert np.array_equal(base_tr, same_tr)
        assert np.array_equal(base_te, same_te)

    def test_cluster_range_valid_and_respects_train_size(self, embeddings_2d, labels):
        train, test = cluster_split(
            embeddings_2d, strategy="subset_sum", y=labels,
            cluster_range=(3, 15), train_size=0.7,
        )
        assert_valid_split(train, test, len(embeddings_2d), ratio_tol=0.4)
        # The search should land close to the 30-example target test size.
        assert abs(len(test) - 30) <= 10

    def test_cluster_range_beats_or_matches_worst_fixed_k(self, embeddings_2d, labels):
        """The searched split should be at least as close to target as some fixed k."""
        _, te_search = cluster_split(
            embeddings_2d, strategy="subset_sum", y=labels, cluster_range=(3, 15)
        )
        _, te_fixed = cluster_split(
            embeddings_2d, strategy="subset_sum", y=labels, n_clusters=3
        )
        target = 30
        assert abs(len(te_search) - target) <= abs(len(te_fixed) - target) + 10

    def test_cluster_range_requires_kmeans(self, embeddings_2d):
        with pytest.raises(ValueError, match="cluster_range requires method='kmeans'"):
            cluster_split(
                embeddings_2d, method="dbscan", strategy="size",
                cluster_range=(3, 10), eps=2.0, min_samples=3,
            )

    def test_cluster_range_bad_pair_raises(self, embeddings_2d):
        with pytest.raises(ValueError, match="cluster_range must be"):
            cluster_split(embeddings_2d, strategy="size", cluster_range=(10, 3))

    def test_fill_individual_exact_test_size(self, embeddings_2d, labels):
        n = len(embeddings_2d)
        target_test = n - int(round(0.7 * n))
        train, test = cluster_split(
            embeddings_2d, n_clusters=8, strategy="closest",
            y=labels, fill_individual=True,
        )
        assert_valid_split(train, test, n, ratio_tol=0.4)
        assert len(test) == target_test

    def test_fill_individual_grows_test_set(self, embeddings_2d, labels):
        """Filling should top up (never shrink) the whole-cluster pocket."""
        _, te_nofill = cluster_split(
            embeddings_2d, n_clusters=8, strategy="closest", random_state=42
        )
        _, te_fill = cluster_split(
            embeddings_2d, n_clusters=8, strategy="closest", random_state=42,
            fill_individual=True,
        )
        assert len(te_fill) >= len(te_nofill)

    def test_cluster_range_and_fill_deterministic(self, embeddings_2d, labels):
        a = cluster_split(
            embeddings_2d, strategy="closest", y=labels,
            cluster_range=(3, 12), fill_individual=True, random_state=7,
        )
        b = cluster_split(
            embeddings_2d, strategy="closest", y=labels,
            cluster_range=(3, 12), fill_individual=True, random_state=7,
        )
        assert np.array_equal(a[0], b[0])
        assert np.array_equal(a[1], b[1])


class TestClusterKFold:
    """Challenging clustering-based cross-validation folds."""

    @pytest.fixture
    def labels(self):
        return np.array([0, 1] * 50)

    def test_returns_valid_fold_ids(self, embeddings_2d, labels):
        folds = cluster_kfold(embeddings_2d, labels, n_folds=5)
        assert folds.shape == (len(embeddings_2d),)
        assert set(folds.tolist()) <= set(range(5))

    def test_all_folds_nonempty(self, embeddings_2d, labels):
        folds = cluster_kfold(embeddings_2d, labels, n_folds=5)
        assert set(folds.tolist()) == set(range(5))

    def test_folds_partition_via_predefined_split(self, embeddings_2d, labels):
        from sklearn.model_selection import PredefinedSplit

        folds = cluster_kfold(embeddings_2d, labels, n_folds=5)
        ps = PredefinedSplit(folds)
        assert ps.get_n_splits() == 5
        seen: set[int] = set()
        for _, test_idx in ps.split():
            assert not (seen & set(test_idx.tolist())), "test folds overlap"
            seen |= set(test_idx.tolist())
        assert seen == set(range(len(embeddings_2d))), "folds don't cover all samples"

    def test_folds_are_label_balanced(self, embeddings_2d, labels):
        folds = cluster_kfold(embeddings_2d, labels, n_folds=5)
        global_p = labels.mean()
        for k in range(5):
            fold_p = labels[folds == k].mean()
            assert abs(fold_p - global_p) < 0.2

    def test_deterministic(self, embeddings_2d, labels):
        a = cluster_kfold(embeddings_2d, labels, n_folds=4, random_state=1)
        b = cluster_kfold(embeddings_2d, labels, n_folds=4, random_state=1)
        assert np.array_equal(a, b)

    def test_y_length_mismatch_raises(self, embeddings_2d):
        with pytest.raises(ValueError, match="length"):
            cluster_kfold(embeddings_2d, np.array([0, 1, 0]), n_folds=3)

    def test_bad_n_folds_raises(self, embeddings_2d, labels):
        with pytest.raises(ValueError, match="n_folds"):
            cluster_kfold(embeddings_2d, labels, n_folds=1)

    def test_too_few_clusters_raises(self, embeddings_2d, labels):
        with pytest.raises(ValueError, match="at least"):
            cluster_kfold(embeddings_2d, labels, n_folds=5, n_clusters=3)

    def test_dbscan_too_few_clusters_raises(self):
        """DBSCAN picks its own cluster count; when it finds fewer than n_folds
        (one dense blob + a noise group) folds would be left empty -> raise."""
        blob = np.random.RandomState(0).randn(60, 2) * 0.05
        noise = np.random.RandomState(99).randn(15, 2) * 30
        X = np.vstack([blob, noise])
        y = np.arange(len(X)) % 2
        with pytest.raises(ValueError, match="fewer than"):
            cluster_kfold(X, y, n_folds=4, method="dbscan", eps=0.5, min_samples=5)

    def test_unknown_method_raises(self, embeddings_2d, labels):
        with pytest.raises(ValueError, match="Unknown clustering method"):
            cluster_kfold(embeddings_2d, labels, method="nope")


class TestClusterKFoldSDS:
    """SDS K-means folds (Wecker, Friedrich & Adel, 2020)."""

    @pytest.fixture
    def labels(self):
        return np.array([0, 1] * 50)

    def test_returns_valid_fold_ids(self, embeddings_2d, labels):
        folds = cluster_kfold(embeddings_2d, labels, n_folds=5, method="sds_kmeans")
        assert folds.shape == (len(embeddings_2d),)
        assert set(folds.tolist()) == set(range(5))

    def test_folds_are_equal_size(self, embeddings_2d, labels):
        # Capacities are chosen so cluster sizes are exactly balanced when n
        # divides evenly; otherwise they differ by at most one.
        folds = cluster_kfold(embeddings_2d, labels, n_folds=5, method="sds_kmeans")
        sizes = np.bincount(folds, minlength=5)
        assert sizes.max() - sizes.min() <= 1
        assert sizes.sum() == len(embeddings_2d)

    def test_uneven_split_sizes_within_one(self, embeddings_2d, labels):
        folds = cluster_kfold(embeddings_2d, labels, n_folds=4, method="sds_kmeans")
        sizes = np.bincount(folds, minlength=4)
        # 100 / 4 = 25 exactly.
        assert sizes.max() - sizes.min() <= 1

    def test_folds_track_global_label_distribution(self, embeddings_2d):
        # Imbalanced, multi-class labels: each fold's per-label counts should be
        # within one of count/n_folds.
        y = np.array([0] * 60 + [1] * 30 + [2] * 10)
        rng = np.random.RandomState(0)
        y = y[rng.permutation(len(y))]
        n_folds = 4
        folds = cluster_kfold(embeddings_2d, y, n_folds=n_folds, method="sds_kmeans")
        for c in np.unique(y):
            total = int((y == c).sum())
            per_fold = [int(((folds == k) & (y == c)).sum()) for k in range(n_folds)]
            assert max(per_fold) - min(per_fold) <= 1
            assert sum(per_fold) == total

    def test_deterministic(self, embeddings_2d, labels):
        a = cluster_kfold(
            embeddings_2d, labels, n_folds=4, method="sds_kmeans", random_state=7
        )
        b = cluster_kfold(
            embeddings_2d, labels, n_folds=4, method="sds_kmeans", random_state=7
        )
        assert np.array_equal(a, b)

    def test_partition_via_predefined_split(self, embeddings_2d, labels):
        from sklearn.model_selection import PredefinedSplit

        folds = cluster_kfold(embeddings_2d, labels, n_folds=5, method="sds_kmeans")
        ps = PredefinedSplit(folds)
        assert ps.get_n_splits() == 5
        seen: set[int] = set()
        for _, test_idx in ps.split():
            assert not (seen & set(test_idx.tolist())), "test folds overlap"
            seen |= set(test_idx.tolist())
        assert seen == set(range(len(embeddings_2d)))

    def test_n_clusters_is_ignored(self, embeddings_2d, labels):
        # sds_kmeans always uses exactly n_folds clusters, so a small n_clusters
        # (which the greedy path rejects) is simply ignored.
        folds = cluster_kfold(
            embeddings_2d, labels, n_folds=5, n_clusters=3, method="sds_kmeans"
        )
        assert set(folds.tolist()) == set(range(5))

    def test_rejects_unknown_kwarg(self, embeddings_2d, labels):
        with pytest.raises(TypeError, match="unexpected keyword"):
            cluster_kfold(
                embeddings_2d, labels, method="sds_kmeans", eps=0.5
            )

    def test_swap_round_improves_partition(self, monkeypatch):
        """The 1-on-1 swap update must actually refine the initial assignment.

        On overlapping (non-separated) Gaussian data the capacity-constrained
        first assignment is suboptimal, and the swap rounds should strictly
        lower the inertia objective. Guard against the swap step silently
        becoming a no-op: run the full pipeline once as-is and once with
        _sds_swap_round patched to do nothing, and require the with-swaps
        partition to differ and score strictly better.
        """
        import splytters.adversarial as adv

        rng = np.random.RandomState(3)
        # Two heavily overlapping blobs: plenty of points end up closer to
        # another fold's centroid after the capacity-constrained first
        # assignment, so swaps have work to do.
        X = np.vstack(
            [rng.randn(60, 2) + [0.5, 0], rng.randn(60, 2) + [-0.5, 0]]
        )
        y = np.array([0, 1] * 60)
        n_folds = 4

        with_swaps = cluster_kfold(
            X, y, n_folds=n_folds, method="sds_kmeans", random_state=11
        )

        def no_swap(points, centroids, assign):
            return assign, 0

        monkeypatch.setattr(adv, "_sds_swap_round", no_swap)
        without_swaps = cluster_kfold(
            X, y, n_folds=n_folds, method="sds_kmeans", random_state=11
        )
        monkeypatch.undo()

        assert not np.array_equal(with_swaps, without_swaps), (
            "swap round had no effect on the returned partition"
        )

        def inertia_of(assign):
            a = assign.astype(np.int64)
            centers = adv._sds_update_centers(X, a, n_folds)
            return adv._sds_inertia(X, a, centers, n_folds)

        assert inertia_of(with_swaps) < inertia_of(without_swaps), (
            "swap rounds should strictly lower cluster-internal variance"
        )


class TestMinoritySplit:
    """Bias-amplified split via per-cluster minority labels (Reif & Schwartz, 2023)."""

    @pytest.fixture
    def biased_data(self):
        """Two well-separated blobs, each dominated by one label with a few
        minority-label instances planted in."""
        rng = np.random.RandomState(0)
        a = rng.randn(20, 2) * 0.2 + np.array([5, 5])
        ya = np.array([0] * 18 + [1] * 2)  # blob A: majority 0, minority {18,19}
        b = rng.randn(20, 2) * 0.2 + np.array([-5, -5])
        yb = np.array([1] * 18 + [0] * 2)  # blob B: majority 1, minority {38,39}
        X = np.vstack([a, b])
        y = np.concatenate([ya, yb])
        return X, y

    def test_routes_cluster_minority_labels_to_test(self, biased_data):
        X, y = biased_data
        train, test = minority_split(X, y, n_clusters=2, random_state=0)
        assert set(test.tolist()) == {18, 19, 38, 39}
        # A full partition of all samples (no overlap, full cover).
        assert set(train.tolist()) | set(test.tolist()) == set(range(40))
        assert not (set(train.tolist()) & set(test.tolist()))

    def test_deterministic(self, biased_data):
        X, y = biased_data
        a = minority_split(X, y, n_clusters=2, random_state=0)
        b = minority_split(X, y, n_clusters=2, random_state=0)
        assert _splits_equal(a, b)

    def test_label_pure_clusters_raise(self):
        rng = np.random.RandomState(1)
        X = np.vstack([rng.randn(20, 2) + [5, 5], rng.randn(20, 2) + [-5, -5]])
        y = np.zeros(40, dtype=int)  # single label everywhere -> no minority
        with pytest.raises(ValueError, match="no minority examples"):
            minority_split(X, y, n_clusters=2)

    def test_degenerate_tiny_test_warns(self):
        # Two well-separated, nearly label-pure blobs with a single planted
        # minority -> a non-empty but degenerate (1-sample) test set.
        rng = np.random.RandomState(0)
        X = np.vstack([rng.randn(50, 2) * 0.1, rng.randn(50, 2) * 0.1 + [10, 0]])
        y = np.array([0] * 50 + [1] * 50)
        y[0] = 1  # one minority-label point inside the label-0 blob
        with pytest.warns(UserWarning, match="degenerate test set"):
            _, test = minority_split(X, y, n_clusters=2)
        assert len(test) == 1

    def test_y_length_mismatch_raises(self, embeddings_2d):
        with pytest.raises(ValueError, match="length"):
            minority_split(embeddings_2d, np.array([0, 1, 0]))

    def test_unknown_method_raises(self, embeddings_2d):
        y = np.array([0, 1] * 50)
        with pytest.raises(ValueError, match="Unknown clustering method"):
            minority_split(embeddings_2d, y, method="nope")

    @pytest.mark.parametrize("method", ["ward", "deepcluster-lite"])
    def test_alt_methods_route_minorities_to_test(self, biased_data, method):
        X, y = biased_data
        train, test = minority_split(X, y, n_clusters=2, method=method, random_state=0)
        # Same planted minorities land in test regardless of clusterer.
        assert set(test.tolist()) == {18, 19, 38, 39}
        assert set(train.tolist()) | set(test.tolist()) == set(range(40))
        assert not (set(train.tolist()) & set(test.tolist()))

    def test_ward_is_deterministic(self, biased_data):
        X, y = biased_data
        a = minority_split(X, y, n_clusters=2, method="ward")
        b = minority_split(X, y, n_clusters=2, method="ward")
        assert _splits_equal(a, b)

    def test_deepcluster_is_deterministic(self, biased_data):
        X, y = biased_data
        a = minority_split(X, y, n_clusters=2, method="deepcluster-lite", random_state=0)
        b = minority_split(X, y, n_clusters=2, method="deepcluster-lite", random_state=0)
        assert _splits_equal(a, b)

    def test_deepcluster_rejects_extra_cluster_kwargs(self, biased_data):
        X, y = biased_data
        with pytest.raises(ValueError, match="no extra cluster_kwargs"):
            minority_split(X, y, method="deepcluster-lite", init="k-means++")

    def test_unknown_minority_labels_raises(self, biased_data):
        X, y = biased_data
        with pytest.raises(ValueError, match="Unknown minority_labels"):
            minority_split(X, y, minority_labels="nope")

    def test_route_matches_split_on_same_clustering(self, biased_data):
        # minority_route on the clustering minority_split computes internally must
        # reproduce minority_split's own train/test partition exactly.
        from splytters.adversarial import _minority_cluster_labels, minority_route

        X, y = biased_data
        labels = _minority_cluster_labels(X, "kmeans", 2, 0)
        r_tr, r_te = minority_route(labels, y)
        s_tr, s_te = minority_split(X, y, n_clusters=2, method="kmeans", random_state=0)
        assert np.array_equal(r_tr, s_tr)
        assert np.array_equal(r_te, s_te)

    def test_route_validates_length_and_labels(self):
        from splytters.adversarial import minority_route

        with pytest.raises(ValueError, match="length"):
            minority_route(np.array([0, 1]), np.array([0, 1, 0]))
        with pytest.raises(ValueError, match="Unknown minority_labels"):
            minority_route(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]),
                           minority_labels="nope")

    def test_least_only_sends_fewer_to_test_on_many_labels(self):
        # One cluster with a clear majority and several rarer labels of unequal
        # size: all_but_majority sends every non-majority label to test, while
        # least_only sends only the single rarest label (footnote 10).
        rng = np.random.RandomState(0)
        X = rng.randn(40, 2) * 0.1  # one tight blob -> a single cluster
        y = np.array([0] * 30 + [1] * 6 + [2] * 3 + [3] * 1)
        _, te_all = minority_split(X, y, n_clusters=1, minority_labels="all_but_majority")
        _, te_least = minority_split(X, y, n_clusters=1, minority_labels="least_only")
        # all-but-majority routes labels {1,2,3} = 10 instances to test.
        assert len(te_all) == 10
        # least-only routes just label 3 (the single rarest) = 1 instance.
        assert set(y[te_least].tolist()) == {3}
        assert len(te_least) == 1


class TestMinorityGrowSplit:
    """minority_split seed grown to a target test size by proximity."""

    @pytest.fixture
    def biased_data(self):
        rng = np.random.RandomState(0)
        a = rng.randn(20, 2) * 0.2 + np.array([5, 5])
        ya = np.array([0] * 18 + [1] * 2)  # blob A: minorities {18,19}
        b = rng.randn(20, 2) * 0.2 + np.array([-5, -5])
        yb = np.array([1] * 18 + [0] * 2)  # blob B: minorities {38,39}
        return np.vstack([a, b]), np.concatenate([ya, yb])

    def test_grows_to_target_size(self, biased_data):
        X, y = biased_data
        train, test = minority_grow_split(X, y, train_size=0.7, n_clusters=2,
                                          random_state=0)
        assert_valid_split(train, test, len(X))
        # target test = n - int(n * 0.7) = 40 - 28 = 12 (seed of 4 grown to 12).
        assert len(test) == 12

    def test_seed_is_subset_of_test(self, biased_data):
        X, y = biased_data
        # The 4 minority seeds must survive into the grown test set.
        train, test = minority_grow_split(X, y, train_size=0.7, n_clusters=2,
                                          random_state=0)
        assert {18, 19, 38, 39}.issubset(set(test.tolist()))

    def test_oversized_seed_is_subsampled_to_target(self):
        # A many-label single cluster makes 'all_but_majority' seed ~75% of the
        # data -- far past a train_size=0.7 target of 30%. Grow can't shrink, so
        # the seed must be subsampled down to exactly the target.
        rng = np.random.RandomState(0)
        X = rng.randn(40, 2) * 0.1
        y = np.array([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10)
        with warnings.catch_warnings():
            # Majority label 0 is never a minority, so it stays out of test -- the
            # missing-class warning is expected here and not what we're testing.
            warnings.simplefilter("ignore")
            train, test = minority_grow_split(X, y, train_size=0.7, n_clusters=1,
                                              random_state=0)
        assert_valid_split(train, test, len(X))
        assert len(test) == 12  # 40 - int(40 * 0.7) = 12, not the ~30-sample seed
        # Round-robin subsampling keeps every class represented in train.
        assert len(np.unique(y[train])) == 4

    def test_minority_labels_passthrough(self, biased_data):
        X, y = biased_data
        # least_only forwards to the seed: the split stays valid and on-target.
        train, test = minority_grow_split(X, y, train_size=0.7, n_clusters=2,
                                          minority_labels="least_only", random_state=0)
        assert_valid_split(train, test, len(X))
        assert len(test) == 12

    def test_invalid_minority_labels_raises(self, biased_data):
        X, y = biased_data
        with pytest.raises(ValueError, match="Unknown minority_labels"):
            minority_grow_split(X, y, minority_labels="nope")

    def test_deterministic(self, biased_data):
        X, y = biased_data
        a = minority_grow_split(X, y, n_clusters=2, random_state=0)
        b = minority_grow_split(X, y, n_clusters=2, random_state=0)
        assert _splits_equal(a, b)

    def test_y_length_mismatch_raises(self, embeddings_2d):
        with pytest.raises(ValueError, match="length"):
            minority_grow_split(embeddings_2d, np.array([0, 1, 0]))

    def test_invalid_stratify_raises(self, embeddings_2d):
        with pytest.raises(ValueError, match="Unknown stratify"):
            minority_grow_split(embeddings_2d, np.zeros(len(embeddings_2d)),
                                stratify="bogus")

    @pytest.mark.parametrize("mode", ["global", "per_class"])
    def test_stratify_balances_test_labels(self, mode):
        # On moons (50/50 labels) proximity-only growth expands into one region
        # and skews the test labels hard; both stratified modes pull it to ~50/50.
        from sklearn.datasets import make_moons

        X, y = make_moons(n_samples=400, noise=0.18, random_state=0)
        _, te_plain = minority_grow_split(X, y, train_size=0.7, n_clusters=8,
                                          random_state=0, stratify="none")
        _, te_strat = minority_grow_split(X, y, train_size=0.7, n_clusters=8,
                                          random_state=0, stratify=mode)
        # Stratified test set mirrors the 50/50 data split (target test = 120 ->
        # 60/60); the un-stratified one is far more lopsided.
        skew_strat = abs(int((y[te_strat] == 0).sum()) - len(te_strat) / 2)
        skew_plain = abs(int((y[te_plain] == 0).sum()) - len(te_plain) / 2)
        assert skew_strat <= 1  # within rounding of perfectly balanced
        assert skew_plain > skew_strat

    @pytest.mark.parametrize("mode", ["global", "per_class"])
    def test_stratify_keeps_seed_and_valid(self, biased_data, mode):
        X, y = biased_data
        train, test = minority_grow_split(X, y, train_size=0.7, n_clusters=2,
                                          random_state=0, stratify=mode)
        assert_valid_split(train, test, len(X))
        assert {18, 19, 38, 39}.issubset(set(test.tolist()))

    @pytest.fixture
    def far_class_data(self):
        # Classes 0/1 share a near-origin cluster (seed + growth stay here); class 2
        # is a far, label-pure blob that proximity growth can't reach with a small
        # test target -> class 2 ends up with no test samples.
        rng = np.random.RandomState(0)
        near = rng.randn(25, 2) * 0.2
        far = rng.randn(30, 2) * 0.2 + np.array([100, 100])
        X = np.vstack([near, far])
        y = np.array([0] * 20 + [1] * 5 + [2] * 30)
        return X, y

    def test_warns_on_class_missing_from_test(self, far_class_data):
        X, y = far_class_data
        with pytest.warns(UserWarning, match=r"no samples for class\(es\) \[2\]"):
            train, test = minority_grow_split(X, y, train_size=0.7, n_clusters=2,
                                              random_state=0, stratify="none")
        assert 2 not in y[np.asarray(test)]

    @pytest.mark.parametrize("mode", ["global", "per_class"])
    def test_stratify_avoids_missing_class(self, far_class_data, mode):
        X, y = far_class_data
        # With quotas, growth reaches the far class (global: forced once near ones
        # fill; per_class: it gets its own anchored neighborhood), so every class
        # appears in test and no warning is raised.
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any UserWarning would fail the test
            train, test = minority_grow_split(X, y, train_size=0.7, n_clusters=2,
                                              random_state=0, stratify=mode)
        assert set(y[np.asarray(test)].tolist()) == {0, 1, 2}


class TestClassBoundarySplit:
    """Class-stratified boundary split: per-class confusable samples -> test."""

    @pytest.fixture
    def two_lines(self):
        """Two classes on a line; within each, the samples nearest the *other*
        class are the boundary samples that should land in test.

        class 0: x = 0..9   (indices 0-9; large x = near class 1)
        class 1: x = 11..20 (indices 10-19; small x = near class 0)
        With train_size=0.7 each class sends its 3 nearest-the-boundary samples
        to test: {7,8,9} from class 0 and {10,11,12} from class 1.
        """
        x = np.concatenate([np.arange(0, 10), np.arange(11, 21)]).astype(float)
        X = np.column_stack([x, np.zeros_like(x)])
        y = np.array([0] * 10 + [1] * 10)
        return X, y

    def test_routes_boundary_samples_to_test(self, two_lines):
        X, y = two_lines
        train, test = class_boundary_split(X, y, train_size=0.7, reference="centroids")
        assert_valid_split(train, test, len(X))
        assert set(test.tolist()) == {7, 8, 9, 10, 11, 12}

    def test_test_set_is_stratified(self, two_lines):
        X, y = two_lines
        _, test = class_boundary_split(X, y, train_size=0.7)
        # Each class contributes the same number of samples to test.
        per_class = [int((y[test] == k).sum()) for k in (0, 1)]
        assert per_class == [3, 3]

    def test_samples_reference_valid(self, two_lines):
        X, y = two_lines
        train, test = class_boundary_split(X, y, train_size=0.7, reference="samples")
        assert_valid_split(train, test, len(X))
        # Same geometry -> same boundary samples as the centroid reference.
        assert set(test.tolist()) == {7, 8, 9, 10, 11, 12}

    def test_deterministic(self, two_lines):
        X, y = two_lines
        a = class_boundary_split(X, y, random_state=0)
        b = class_boundary_split(X, y, random_state=7)  # ignored; deterministic
        assert _splits_equal(a, b)

    def test_single_class_raises(self, two_lines):
        X, _ = two_lines
        with pytest.raises(ValueError, match="at least 2 distinct classes"):
            class_boundary_split(X, np.zeros(len(X), dtype=int))

    def test_y_length_mismatch_raises(self, embeddings_2d):
        with pytest.raises(ValueError, match="length"):
            class_boundary_split(embeddings_2d, np.array([0, 1, 0]))

    def test_unknown_reference_raises(self, two_lines):
        X, y = two_lines
        with pytest.raises(ValueError, match="reference must be"):
            class_boundary_split(X, y, reference="nope")


class TestDecisionBoundarySplit:
    """Supervised learned-boundary split: lowest-margin samples -> test."""

    @pytest.fixture
    def two_lines(self):
        """Binary, separable along x; samples nearest x=10 are hardest."""
        x = np.concatenate([np.arange(0, 10), np.arange(11, 21)]).astype(float)
        X = np.column_stack([x, np.zeros_like(x)])
        y = np.array([0] * 10 + [1] * 10)
        return X, y

    @pytest.fixture
    def three_blobs(self):
        rng = np.random.RandomState(0)
        X = np.vstack([
            rng.randn(30, 5) + np.array([4, 0, 0, 0, 0]),
            rng.randn(30, 5) + np.array([0, 4, 0, 0, 0]),
            rng.randn(30, 5) + np.array([0, 0, 4, 0, 0]),
        ])
        y = np.array([0] * 30 + [1] * 30 + [2] * 30)
        return X, y

    @pytest.mark.parametrize("model", ["linear_svc", "logistic", "rbf_svc"])
    @pytest.mark.parametrize("stratify", ["per_class", "global"])
    def test_valid_split_binary(self, two_lines, model, stratify):
        X, y = two_lines
        train, test = decision_boundary_split(
            X, y, model=model, stratify=stratify, random_state=0
        )
        assert_valid_split(train, test, len(X), ratio_tol=0.1)

    @pytest.mark.parametrize("model", ["linear_svc", "logistic", "rbf_svc"])
    def test_valid_split_multiclass(self, three_blobs, model):
        X, y = three_blobs
        train, test = decision_boundary_split(X, y, model=model, random_state=0)
        assert_valid_split(train, test, len(X), ratio_tol=0.1)

    def test_rbf_concentrates_on_nonlinear_boundary(self):
        """On concentric circles (no linear boundary), the RBF surrogate routes
        test points tightly onto the true circular boundary, where the linear
        surrogate -- which has no meaningful boundary -- scatters them."""
        from sklearn.datasets import make_circles

        X, y = make_circles(n_samples=400, noise=0.08, factor=0.5, random_state=0)
        radii = {}
        for model in ("linear_svc", "rbf_svc"):
            _, test = decision_boundary_split(X, y, model=model, random_state=0)
            radii[model] = np.sqrt((X[test] ** 2).sum(axis=1)).std()
        assert radii["rbf_svc"] < radii["linear_svc"]

    @pytest.mark.parametrize("model", ["linear_svc", "logistic"])
    def test_boundary_samples_go_to_test(self, two_lines, model):
        """The samples nearest the learned boundary (x=10) are the hardest."""
        X, y = two_lines
        _, test = decision_boundary_split(X, y, train_size=0.7, model=model)
        assert set(test.tolist()) == {7, 8, 9, 10, 11, 12}

    def test_per_class_is_label_balanced(self, three_blobs):
        X, y = three_blobs
        _, test = decision_boundary_split(X, y, train_size=0.7, stratify="per_class")
        assert [int((y[test] == k).sum()) for k in (0, 1, 2)] == [9, 9, 9]

    def test_entropy_score_with_logistic(self, three_blobs):
        X, y = three_blobs
        train, test = decision_boundary_split(
            X, y, model="logistic", score="entropy", random_state=0
        )
        assert_valid_split(train, test, len(X), ratio_tol=0.1)

    def test_deterministic(self, three_blobs):
        X, y = three_blobs
        assert _splits_equal(
            decision_boundary_split(X, y, random_state=1),
            decision_boundary_split(X, y, random_state=1),
        )

    @pytest.mark.parametrize("model", ["linear_svc", "rbf_svc"])
    def test_entropy_requires_logistic(self, two_lines, model):
        X, y = two_lines
        with pytest.raises(ValueError, match="entropy"):
            decision_boundary_split(X, y, model=model, score="entropy")

    def test_single_class_raises(self, two_lines):
        X, _ = two_lines
        with pytest.raises(ValueError, match="at least 2"):
            decision_boundary_split(X, np.zeros(len(X), dtype=int))

    def test_y_length_mismatch_raises(self, two_lines):
        X, y = two_lines
        with pytest.raises(ValueError, match="length"):
            decision_boundary_split(X, y[:-1])

    def test_class_too_small_to_hold_out_raises(self):
        X = np.random.RandomState(0).randn(11, 3)
        y = np.array([0] * 10 + [1])  # class 1 has a single sample
        with pytest.raises(ValueError, match="2 samples"):
            decision_boundary_split(X, y)

    @pytest.mark.parametrize("kw", [
        {"model": "x"}, {"score": "x"}, {"stratify": "x"},
    ])
    def test_unknown_options_raise(self, two_lines, kw):
        X, y = two_lines
        with pytest.raises(ValueError):
            decision_boundary_split(X, y, **kw)


class TestMaximinSplit:
    """Farthest-point (k-center) test selection — a diverse, spread-out test."""

    def test_valid_split(self, embeddings_2d):
        train, test = maximin_split(embeddings_2d, train_size=0.7)
        assert_valid_split(train, test, len(embeddings_2d), ratio_tol=0.05)

    def test_test_more_spread_than_random(self, embeddings_2d):
        """The farthest-point test set is more diverse (larger mean pairwise
        distance) than a random test set of the same size."""
        from scipy.spatial.distance import pdist
        _, m_te = maximin_split(embeddings_2d, train_size=0.7, random_state=0)
        _, r_te = random_split(embeddings_2d, train_size=0.7, random_state=0)
        assert pdist(embeddings_2d[m_te]).mean() > pdist(embeddings_2d[r_te]).mean()

    def test_deterministic(self, embeddings_2d):
        assert _splits_equal(
            maximin_split(embeddings_2d, random_state=1),
            maximin_split(embeddings_2d, random_state=1),
        )


class TestMMDMaximizedSplit:
    """Adversarial dual of mmd_minimized_split (Napoli & White, 2025)."""

    @staticmethod
    def _mmd(emb, train, test):
        gamma = 1.0 / emb.shape[1]
        from scipy.spatial.distance import cdist

        def k(a, b):
            return np.exp(-gamma * cdist(a, b, "sqeuclidean"))

        T, S = emb[train], emb[test]
        m, n = len(T), len(S)
        return (
            k(T, T).sum() / (m * m)
            + k(S, S).sum() / (n * n)
            - 2 * k(T, S).sum() / (m * n)
        )

    def test_valid_split(self, embeddings_2d):
        train, test = mmd_maximized_split(embeddings_2d)
        assert_valid_split(train, test, len(embeddings_2d))

    def test_higher_mmd_than_minimized(self, embeddings_2d):
        tr_max, te_max = mmd_maximized_split(embeddings_2d, n_iterations=500)
        tr_min, te_min = mmd_minimized_split(embeddings_2d, n_iterations=500)
        assert self._mmd(embeddings_2d, tr_max, te_max) > self._mmd(
            embeddings_2d, tr_min, te_min
        )

    def test_deterministic(self, embeddings_2d):
        a = mmd_maximized_split(embeddings_2d, random_state=3)
        b = mmd_maximized_split(embeddings_2d, random_state=3)
        assert _splits_equal(a, b)

    def test_invalid_kernel_raises(self, embeddings_2d):
        with pytest.raises(ValueError, match="kernel must be"):
            mmd_maximized_split(embeddings_2d, kernel="bogus")


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

    def test_spectral_seed_independent(self):
        """The Fiedler sign is oriented deterministically, so different seeds
        give the identical split (previously the sign flip yielded disjoint
        held-out sets)."""
        rng = np.random.RandomState(0)
        X = np.vstack([rng.randn(50, 10), rng.randn(50, 10) + 4])
        a = min_cut_split(X, train_size=0.8, method="spectral", random_state=0)
        b = min_cut_split(X, train_size=0.8, method="spectral", random_state=1)
        assert _splits_equal(a, b)

    def test_tiny_dataset(self):
        X = np.array([[0, 0], [1, 1]])
        train, test = min_cut_split(X, train_size=0.5)
        assert_valid_split(train, test, 2, train_size=0.5, ratio_tol=0.5)

    def test_invalid_method(self, embeddings_small):
        with pytest.raises(ValueError, match="Unknown method"):
            min_cut_split(embeddings_small, method="invalid")

    def test_spectral_eig_failure_falls_back_to_random(
        self, embeddings_small, monkeypatch
    ):
        """If eigendecomposition raises, the split falls back to a random one
        and warns that the result is not adversarial."""
        def boom(*args, **kwargs):
            raise RuntimeError("eig failed")
        monkeypatch.setattr("splytters.adversarial.dense_eigh", boom)
        with pytest.warns(UserWarning, match="NOT adversarial"):
            train, test = min_cut_split(embeddings_small, method="spectral")
        assert_valid_split(train, test, len(embeddings_small), ratio_tol=0.3)

    def test_spectral_illconditioned_graph_terminates(self):
        """Regression: a large, tightly-clustered graph with near-duplicate rows
        (the covertype-like case that made sparse ARPACK ``eigsh(which='SM')``
        hang for hours) must now solve via the dense partial eigendecomposition —
        it returns a real adversarial split, without the random-fallback warning."""
        rng = np.random.RandomState(0)
        X = np.vstack([rng.randn(300, 12) + c * 5 for c in range(6)])  # 1800 pts
        X += rng.randn(*X.shape) * 1e-6  # near-duplicates stress the solver
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a fallback warning would raise here
            train, test = min_cut_split(X, train_size=0.7, method="spectral")
        assert_valid_split(train, test, len(X), train_size=0.7, ratio_tol=0.05)

    def test_stoer_wagner_connected_graph(self, embeddings_small):
        """A zero threshold keeps all edges -> one connected component, so the
        exact Stoer-Wagner partition path runs."""
        pytest.importorskip("networkx")
        train, test = min_cut_split(
            embeddings_small, method="stoer_wagner", similarity_threshold=0.0
        )
        assert_valid_split(train, test, len(embeddings_small), ratio_tol=0.4)

    def test_stoer_wagner_connected_small_train(self, embeddings_small):
        """A small train_size makes the cut's first partition large enough to
        fill train directly (the `len(set1) >= n_train` branch)."""
        pytest.importorskip("networkx")
        train, test = min_cut_split(
            embeddings_small, train_size=0.1,
            method="stoer_wagner", similarity_threshold=0.0,
        )
        assert_valid_split(
            train, test, len(embeddings_small), train_size=0.1, ratio_tol=0.5,
        )

    def test_stoer_wagner_disconnected_graph(self):
        """Two tight, far-apart clusters with a high similarity threshold yield
        a disconnected graph, exercising the connected-components branch."""
        pytest.importorskip("networkx")
        rng = np.random.RandomState(0)
        a = rng.randn(10, 2) * 0.1 + np.array([0, 0])
        b = rng.randn(10, 2) * 0.1 + np.array([100, 100])
        X = np.vstack([a, b])
        train, test = min_cut_split(
            X, method="stoer_wagner", similarity_threshold=0.9
        )
        assert_valid_split(train, test, len(X), ratio_tol=0.5)

    def test_stoer_wagner_missing_networkx_raises(self, embeddings_small, monkeypatch):
        """Without networkx, the stoer_wagner method raises a clear ImportError."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "networkx":
                raise ImportError("no networkx")
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="networkx is required"):
            min_cut_split(embeddings_small, method="stoer_wagner")


class TestNormalizedCutSplit:

    def test_valid_split(self, embeddings_small):
        train, test = normalized_cut_split(embeddings_small)
        assert_valid_split(train, test, len(embeddings_small), ratio_tol=0.3)

    def test_tiny_dataset(self):
        X = np.array([[0, 0], [1, 1]])
        train, test = normalized_cut_split(X, train_size=0.5)
        assert_valid_split(train, test, 2, train_size=0.5, ratio_tol=0.5)

    def test_deterministic_sign_orientation(self):
        """The Fiedler vector is sign-oriented so the partition is stable. This
        dataset's largest-magnitude Fiedler component is negative, exercising the
        sign-flip branch; the split must be valid and repeatable."""
        X = np.random.RandomState(0).randn(12, 2)
        a = normalized_cut_split(X, train_size=0.5)
        b = normalized_cut_split(X, train_size=0.5)
        assert _splits_equal(a, b)
        assert_valid_split(a[0], a[1], len(X), train_size=0.5, ratio_tol=0.2)


class TestWassersteinAdversarialSplit:

    def test_valid_split(self, embeddings_2d):
        train, test = wasserstein_adversarial_split(embeddings_2d)
        assert_valid_split(train, test, len(embeddings_2d), ratio_tol=0.2)

    def test_deterministic(self, embeddings_2d):
        a = wasserstein_adversarial_split(embeddings_2d, random_state=3)
        b = wasserstein_adversarial_split(embeddings_2d, random_state=3)
        assert _splits_equal(a, b)


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
        train, test = stratified_random_split(embeddings_2d, labels, train_size=0.8)
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

    def test_invalid_kernel_raises(self, embeddings_2d):
        with pytest.raises(ValueError, match="kernel must be"):
            mmd_minimized_split(embeddings_2d, kernel="bogus")


class TestMmdKernelKmeansMethod:
    """The paper-faithful constrained kernel k-means / LP method of
    Napoli & White (TMLR 2025; arXiv 2024), method="kernel_kmeans"."""

    @staticmethod
    def _mmd(emb, train, test):
        return TestMMDMaximizedSplit._mmd(emb, train, test)

    @staticmethod
    def _labels(n):
        return np.array([i % 2 for i in range(n)])

    def test_maximized_valid_split(self, embeddings_2d):
        train, test = mmd_maximized_split(embeddings_2d, method="kernel_kmeans")
        assert_valid_split(train, test, len(embeddings_2d))

    def test_minimized_valid_split(self, embeddings_2d):
        train, test = mmd_minimized_split(embeddings_2d, method="kernel_kmeans")
        assert_valid_split(train, test, len(embeddings_2d))

    def test_respects_absolute_train_size(self, embeddings_2d):
        train, test = mmd_maximized_split(
            embeddings_2d, train_size=60, method="kernel_kmeans"
        )
        assert len(train) == 60
        assert len(test) == len(embeddings_2d) - 60

    def test_maximized_beats_random(self, embeddings_2d):
        # On a synthetic two-cluster dataset, the kernel k-means (k=2) split
        # should reach a higher MMD than a random split.
        train, test = mmd_maximized_split(embeddings_2d, method="kernel_kmeans")
        achieved = self._mmd(embeddings_2d, train, test)

        rng = np.random.RandomState(0)
        n = len(embeddings_2d)
        n_train = int(0.7 * n)
        random_mmds = []
        for _ in range(20):
            idx = rng.permutation(n)
            random_mmds.append(
                self._mmd(embeddings_2d, idx[:n_train], idx[n_train:])
            )
        assert achieved >= max(random_mmds)

    def test_maximized_exceeds_minimized(self, embeddings_2d):
        # On the same data, kernel, and seed, the max split must reach a
        # strictly higher MMD than the min split.
        tr_max, te_max = mmd_maximized_split(
            embeddings_2d, method="kernel_kmeans", random_state=3
        )
        tr_min, te_min = mmd_minimized_split(
            embeddings_2d, method="kernel_kmeans", random_state=3
        )
        assert self._mmd(embeddings_2d, tr_max, te_max) > self._mmd(
            embeddings_2d, tr_min, te_min
        )

    def test_lp_objective_sign(self, embeddings_2d, monkeypatch):
        # Kill-check for an LP objective sign flip. End-to-end MMD comparisons
        # cannot catch it on separable data (a wrongly-signed full step still
        # oscillates onto segregated, high-MMD vertices and best-partition
        # tracking keeps them), so assert the objective coefficients directly:
        # kernel k-means minimizes the scatter, so the max-MMD path must pass
        # the *nonnegative* squared kernel distances D as minimization costs,
        # and the min-MMD dual must pass their negation.
        import scipy.optimize

        real_linprog = scipy.optimize.linprog
        costs = []

        def spy_linprog(c, *args, **kwargs):
            costs.append(np.asarray(c).copy())
            return real_linprog(c, *args, **kwargs)

        monkeypatch.setattr(scipy.optimize, "linprog", spy_linprog)

        mmd_maximized_split(embeddings_2d, method="kernel_kmeans")
        assert costs, "max path never reached the LP"
        for c in costs:
            assert np.all(c >= -1e-8), "max-MMD path must minimize +D"
            assert c.max() > 1e-6  # distances are not degenerate

        costs.clear()
        mmd_minimized_split(embeddings_2d, method="kernel_kmeans")
        assert costs, "min path never reached the LP"
        for c in costs:
            assert np.all(c <= 1e-8), "min-MMD dual must maximize D (pass -D)"
            assert c.min() < -1e-6

    def test_lp_solution_respects_group_masses(self, embeddings_2d, monkeypatch):
        # Exercise the LP constraints (paper Eq. 15) independently of the
        # per-group rounding: capture every raw linprog solution and check that
        # its per-group validation mass is exactly the apportioned target.
        import scipy.optimize

        real_linprog = scipy.optimize.linprog
        solutions = []

        def spy_linprog(c, *args, **kwargs):
            res = real_linprog(c, *args, **kwargs)
            solutions.append(res.x.copy())
            return res

        monkeypatch.setattr(scipy.optimize, "linprog", spy_linprog)

        n = len(embeddings_2d)
        y = self._labels(n)
        groups = np.array([i % 5 for i in range(n)])
        train, test = mmd_maximized_split(
            embeddings_2d, method="kernel_kmeans", y=y, groups=groups
        )
        assert len(solutions) > 0

        # Reconstruct the per-group validation targets the same way the helper
        # does: largest-remainder apportionment of train slots over Y x D.
        from splytters.utils import apportion_train

        keys = [(y[i], groups[i]) for i in range(n)]
        unique_keys = sorted(set(keys), key=repr)
        members = {k: [i for i in range(n) if keys[i] == k] for k in unique_keys}
        sizes = [len(members[k]) for k in unique_keys]
        per_group_train = apportion_train(sizes, int(0.7 * n))
        val_targets = {
            k: sz - tr
            for k, sz, tr in zip(unique_keys, sizes, per_group_train, strict=True)
        }

        for x in solutions:
            U = x.reshape(n, 2)
            # Every point assigned exactly once (Eq. 14).
            np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-8)
            # Per-group validation mass hits the target exactly (Eq. 15).
            for k in unique_keys:
                mass = U[members[k], 1].sum()
                assert abs(mass - val_targets[k]) < 1e-8, (
                    f"group {k}: LP val mass {mass} != target {val_targets[k]}"
                )

        # Rounded output matches the same targets (sanity tie-in).
        for k in unique_keys:
            in_test = sum(1 for i in test if keys[i] == k)
            assert in_test == val_targets[k]

    def test_minimized_below_random(self, embeddings_2d):
        train, test = mmd_minimized_split(embeddings_2d, method="kernel_kmeans")
        achieved = self._mmd(embeddings_2d, train, test)
        rng = np.random.RandomState(0)
        n = len(embeddings_2d)
        n_train = int(0.7 * n)
        idx = rng.permutation(n)
        random_mmd = self._mmd(embeddings_2d, idx[:n_train], idx[n_train:])
        assert achieved <= random_mmd

    def test_label_proportions_constrained(self, embeddings_2d):
        y = self._labels(len(embeddings_2d))
        train, test = mmd_maximized_split(
            embeddings_2d, method="kernel_kmeans", y=y
        )
        # Each label's train/test proportion should match the global fraction.
        for label in np.unique(y):
            members = np.sum(y == label)
            in_test = np.sum(y[test] == label)
            assert abs(in_test / members - 0.3) <= 1.0 / members + 1e-9

    def test_group_proportions_constrained(self, embeddings_2d):
        groups = np.array([i % 5 for i in range(len(embeddings_2d))])
        train, test = mmd_minimized_split(
            embeddings_2d, method="kernel_kmeans", groups=groups
        )
        for g in np.unique(groups):
            members = np.sum(groups == g)
            in_test = np.sum(groups[test] == g)
            assert abs(in_test / members - 0.3) <= 1.0 / members + 1e-9

    def test_deterministic(self, embeddings_2d):
        a = mmd_maximized_split(
            embeddings_2d, method="kernel_kmeans", random_state=5
        )
        b = mmd_maximized_split(
            embeddings_2d, method="kernel_kmeans", random_state=5
        )
        assert _splits_equal(a, b)

    def test_minimized_deterministic(self, embeddings_2d):
        a = mmd_minimized_split(
            embeddings_2d, method="kernel_kmeans", random_state=5
        )
        b = mmd_minimized_split(
            embeddings_2d, method="kernel_kmeans", random_state=5
        )
        assert _splits_equal(a, b)

    def test_y_groups_rejected_for_swap(self, embeddings_2d):
        y = self._labels(len(embeddings_2d))
        with pytest.raises(ValueError, match="kernel_kmeans"):
            mmd_maximized_split(embeddings_2d, method="swap", y=y)
        with pytest.raises(ValueError, match="kernel_kmeans"):
            mmd_minimized_split(embeddings_2d, method="swap", groups=y)

    def test_invalid_method_raises(self, embeddings_2d):
        with pytest.raises(ValueError, match="method must be"):
            mmd_maximized_split(embeddings_2d, method="bogus")
        with pytest.raises(ValueError, match="method must be"):
            mmd_minimized_split(embeddings_2d, method="bogus")

    def test_linear_kernel(self, embeddings_2d):
        train, test = mmd_maximized_split(
            embeddings_2d, method="kernel_kmeans", kernel="linear"
        )
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

    def test_every_test_nn_in_train(self, embeddings_2d):
        """The documented contract: *every* test sample's nearest neighbor is
        in train. The old greedy could later pull a point's NN into test."""
        train, test = nearest_neighbor_split(embeddings_2d)
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=2, algorithm="auto")
        nn.fit(embeddings_2d)
        neighbors = nn.kneighbors(embeddings_2d, return_distance=False)[:, 1]

        train_set = set(train.tolist())
        assert all(neighbors[t_idx] in train_set for t_idx in test.tolist())

    def test_warns_when_test_cannot_be_filled(self):
        """When keeping every NN in train prevents reaching n_test, the split
        returns a smaller test set and warns rather than silently under-filling.
        Two tight mutual-NN pairs far apart: each point moved to test pins its
        partner into train, so at most 2 of the 3 requested test points fit."""
        X = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]])
        with pytest.warns(UserWarning, match="could only place"):
            train, test = nearest_neighbor_split(X, train_size=0.25)
        assert_valid_split(train, test, 4, train_size=0.25, ratio_tol=0.3)
        assert len(test) < 3


class TestDuplicateSpreadSplit:

    def test_valid_split(self, embeddings_2d):
        train, test = duplicate_spread_split(embeddings_2d)
        assert_valid_split(train, test, len(embeddings_2d), ratio_tol=0.3)

    def test_singletons_apportioned_to_hit_train_size(self):
        """All-unique points (every group a singleton) must still split near
        train_size; the old version dumped every singleton into train, leaving
        the test set empty."""
        rng = np.random.RandomState(0)
        X = rng.randn(20, 4) * 100
        # A tiny threshold makes every distinct point its own group (all
        # singletons), isolating the singleton-apportionment path.
        train, test = duplicate_spread_split(
            X, train_size=0.7, similarity_threshold=1e-9
        )
        assert_valid_split(train, test, 20, train_size=0.7, ratio_tol=0.1)
        assert len(test) > 0


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
        assert _splits_equal(a, b)

    def test_cluster_split(self, embeddings_2d):
        a = cluster_split(embeddings_2d, random_state=1)
        b = cluster_split(embeddings_2d, random_state=1)
        assert _splits_equal(a, b)

    def test_distribution_matched(self, embeddings_2d):
        a = distribution_matched_split(embeddings_2d, n_iterations=20, random_state=1)
        b = distribution_matched_split(embeddings_2d, n_iterations=20, random_state=1)
        assert _splits_equal(a, b)

    def test_nearest_neighbor(self, embeddings_2d):
        a = nearest_neighbor_split(embeddings_2d, random_state=1)
        b = nearest_neighbor_split(embeddings_2d, random_state=1)
        assert _splits_equal(a, b)


class TestListInput:
    """Splitters should accept plain Python lists, not just numpy arrays."""

    def test_random_split_with_list(self):
        X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        train, test = random_split(X, train_size=0.6)
        assert_valid_split(train, test, 5, train_size=0.6, ratio_tol=0.3)

    def test_cluster_split_with_list(self):
        X = [[i, i] for i in range(20)]
        train, test = cluster_split(X, n_clusters=2)
        assert_valid_split(train, test, 20, ratio_tol=0.3)


# ===========================================================================
# Input validation
# ===========================================================================

class TestValidateSplitInputs:

    def test_ratio_too_high(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_split_inputs(np.zeros((10, 2)), 1.0)

    def test_ratio_too_low(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_split_inputs(np.zeros((10, 2)), 0.0)

    def test_ratio_negative(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_split_inputs(np.zeros((10, 2)), -0.5)

    def test_ratio_above_one(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_split_inputs(np.zeros((10, 2)), 1.5)

    def test_too_few_samples(self):
        with pytest.raises(ValueError, match="at least 2 samples"):
            validate_split_inputs(np.zeros((1, 2)), 0.5)

    def test_empty_array(self):
        with pytest.raises(ValueError, match="at least 2 samples"):
            validate_split_inputs(np.zeros((0, 2)), 0.5)

    def test_valid_passes(self):
        validate_split_inputs(np.zeros((10, 2)), 0.7)  # should not raise


class TestSplitterValidation:
    """Splitters propagate validation errors."""

    def test_random_split_bad_ratio(self):
        X = np.zeros((10, 2))
        with pytest.raises(ValueError, match="between 0 and 1"):
            random_split(X, train_size=1.5)

    def test_cluster_split_bad_ratio(self):
        X = np.zeros((10, 2))
        with pytest.raises(ValueError, match="between 0 and 1"):
            cluster_split(X, train_size=0.0)

    def test_distance_adversarial_too_few(self):
        X = np.zeros((1, 2))
        with pytest.raises(ValueError, match="at least 2 samples"):
            distance_adversarial_split(X)

    def test_moment_matched_bad_ratio(self):
        X = np.zeros((10, 2))
        with pytest.raises(ValueError, match="between 0 and 1"):
            moment_matched_split(X, train_size=-0.1)

    def test_cluster_leak_bad_ratio(self):
        X = np.zeros((10, 2))
        with pytest.raises(ValueError, match="between 0 and 1"):
            cluster_leak_split(X, train_size=1.0)

    def test_nearest_neighbor_too_few(self):
        X = np.zeros((1, 2))
        with pytest.raises(ValueError, match="at least 2 samples"):
            nearest_neighbor_split(X)


# ===========================================================================
# sklearn-aligned conventions (ndarray returns, check_array, train_size)
# ===========================================================================

class TestSklearnAlignment:
    """Returns and input handling should match sklearn conventions."""

    def test_returns_integer_ndarrays(self, embeddings_2d):
        train, test = random_split(embeddings_2d)
        assert isinstance(train, np.ndarray) and isinstance(test, np.ndarray)
        assert np.issubdtype(train.dtype, np.integer)
        assert np.issubdtype(test.dtype, np.integer)

    def test_int_train_size_is_absolute_count(self, embeddings_2d):
        train, test = random_split(embeddings_2d, train_size=30)
        assert len(train) == 30
        assert len(test) == len(embeddings_2d) - 30

    def test_int_train_size_in_splitter(self, embeddings_2d):
        train, test = distance_adversarial_split(embeddings_2d, train_size=40)
        assert len(train) == 40

    def test_rejects_nan(self):
        X = np.ones((10, 3))
        X[0, 0] = np.nan
        with pytest.raises(ValueError):
            random_split(X)

    def test_rejects_inf(self):
        X = np.ones((10, 3))
        X[2, 1] = np.inf
        with pytest.raises(ValueError):
            cluster_split(X, n_clusters=2)

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError):
            random_split(np.arange(10))

    def test_accepts_python_list_returns_ndarray(self):
        X = [[i, i + 1] for i in range(12)]
        train, test = random_split(X, train_size=0.5)
        assert isinstance(train, np.ndarray)
        assert len(train) == 6

    def test_torch_tensor_input(self):
        torch = pytest.importorskip("torch")
        X = torch.randn(20, 4)
        train, test = random_split(X, train_size=0.7)
        assert isinstance(train, np.ndarray)
        assert len(train) + len(test) == 20


# ===========================================================================
# Regression tests for fixed correctness bugs
# ===========================================================================

class TestBinSplitApportionment:
    """Bin/cluster splitters must never empty the test set and must hit the
    requested train fraction (largest-remainder apportionment)."""

    def test_apportion_train_sums_exactly(self):
        counts = apportion_train([4, 4, 4, 4], 10)
        assert counts.sum() == 10
        assert all(0 <= c <= 4 for c in counts)

    def test_apportion_singletons_leave_some_for_test(self):
        # 12 singleton bins, target 8 train -> 8 bins train, 4 test.
        counts = apportion_train([1] * 12, 8)
        assert counts.sum() == 8
        assert (counts == 0).sum() == 4

    @pytest.mark.parametrize("splitter", [
        stratified_similarity_split, density_balanced_split,
    ])
    def test_singleton_bins_do_not_empty_test(self, splitter):
        """n_bins >= n_samples used to force every sample into train."""
        X = np.random.RandomState(0).randn(12, 3)
        train, test = splitter(X, train_size=0.7, n_bins=20)
        assert len(test) > 0
        assert_valid_split(train, test, 12, ratio_tol=0.25)

    @pytest.mark.parametrize("splitter,kw", [
        (stratified_similarity_split, {"n_bins": 10}),
        (density_balanced_split, {"n_bins": 10}),
        (cluster_leak_split, {"n_clusters": 8}),
    ])
    def test_hits_requested_train_fraction(self, splitter, kw):
        """Per-bin int() truncation used to undershoot (got ~50% for 0.7)."""
        X = np.random.RandomState(0).randn(40, 5)
        train, test = splitter(X, train_size=0.7, **kw)
        assert len(train) == 28  # resolve_n_train(40, 0.7)
        assert_valid_split(train, test, 40, ratio_tol=0.05)


class TestResolveNTrainClamp:

    def test_small_fraction_never_empties_a_side(self):
        # int(2 * 0.3) == 0 previously -> empty train.
        assert resolve_n_train(2, 0.3) == 1
        train, test = random_split(np.zeros((2, 4)), train_size=0.3)
        assert_valid_split(train, test, 2, train_size=0.3, ratio_tol=0.5)

    def test_normal_cases_unchanged(self):
        assert resolve_n_train(10, 0.7) == 7
        assert resolve_n_train(100, 0.9) == 90
        assert resolve_n_train(100, 50) == 50  # absolute count passes through

    def test_high_fraction_leaves_one_for_test(self):
        assert resolve_n_train(10, 0.999) == 9


class TestHistogramMatchedConstantDim:

    def test_constant_dimension_does_not_produce_nan(self):
        """A constant feature collapses percentile bin edges; density=True used
        to divide by zero and NaN-poison the optimizer score."""
        import warnings
        X = np.random.RandomState(0).randn(30, 4)
        X[:, 2] = 5.0  # constant dimension
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any RuntimeWarning fails the test
            train, test = histogram_matched_split(X, train_size=0.6, n_iterations=30)
        assert_valid_split(train, test, 30, train_size=0.6, ratio_tol=0.2)
