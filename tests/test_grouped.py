"""Tests for grouping-aware splits (group_split, deduplicated_split)."""

import warnings

import numpy as np
import pytest

from splytters import deduplicated_split, group_split
from splytters.utils import near_duplicate_components


def _valid(train, test, n):
    s_tr, s_te = set(train.tolist()), set(test.tolist())
    return (
        (s_tr | s_te) == set(range(n))
        and not (s_tr & s_te)
        and len(train) > 0
        and len(test) > 0
    )


class TestGroupSplit:

    @pytest.fixture
    def grouped_data(self):
        rng = np.random.RandomState(0)
        X = rng.randn(120, 8)
        groups = np.repeat(np.arange(20), 6)  # 20 groups of 6
        return X, groups

    def test_valid_split(self, grouped_data):
        X, groups = grouped_data
        train, test = group_split(X, groups, train_size=0.7)
        assert _valid(train, test, len(X))

    def test_no_group_spans_both_sides(self, grouped_data):
        X, groups = grouped_data
        train, test = group_split(X, groups, train_size=0.7)
        assert set(groups[train]) & set(groups[test]) == set()

    def test_approximate_train_size(self, grouped_data):
        X, groups = grouped_data
        train, _ = group_split(X, groups, train_size=0.7)
        assert abs(len(train) / len(X) - 0.7) < 0.15

    def test_deterministic(self, grouped_data):
        X, groups = grouped_data
        a = group_split(X, groups, random_state=1)
        b = group_split(X, groups, random_state=1)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

    def test_length_mismatch_raises(self, grouped_data):
        X, groups = grouped_data
        with pytest.raises(ValueError, match="length"):
            group_split(X, groups[:-1])

    def test_single_group_raises(self):
        X = np.random.RandomState(0).randn(10, 3)
        with pytest.raises(ValueError, match="at least 2"):
            group_split(X, np.zeros(10, dtype=int))


class TestDeduplicatedSplit:

    @pytest.fixture
    def dup_data(self):
        """Well-separated bases, each with one near-exact duplicate: pairs
        (i, i + 60) are near-duplicates."""
        rng = np.random.RandomState(0)
        base = rng.randn(60, 8) * 10
        return np.vstack([base, base + 1e-4])

    def test_valid_split(self, dup_data):
        train, test = deduplicated_split(
            dup_data, train_size=0.7, similarity_threshold=0.1
        )
        assert _valid(train, test, len(dup_data))

    def test_no_near_duplicate_pair_split(self, dup_data):
        train, _ = deduplicated_split(
            dup_data, train_size=0.7, similarity_threshold=0.1
        )
        trs = set(train.tolist())
        for i in range(60):  # each (i, i+60) duplicate pair stays together
            assert (i in trs) == ((i + 60) in trs)

    def test_deterministic(self, dup_data):
        a = deduplicated_split(dup_data, similarity_threshold=0.1, random_state=1)
        b = deduplicated_split(dup_data, similarity_threshold=0.1, random_state=1)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

    def test_all_one_component_raises(self):
        """Identical points collapse to a single component — nothing to split."""
        X = np.ones((10, 4))
        with pytest.raises(ValueError, match="one near-duplicate component"):
            deduplicated_split(X)

    def test_default_threshold_respects_train_size(self):
        """Regression for #68: the old default (1st percentile of all pairwise
        distances) merged ~95% of this data into one component, giving a
        209 / 3791 split for train_size=0.8, without a warning."""
        X = np.random.default_rng(0).normal(size=(4000, 64))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            train, test = deduplicated_split(X, train_size=0.8)
        assert _valid(train, test, len(X))
        assert len(train) == 3200

    def test_default_threshold_keeps_injected_duplicates_together(self):
        X, pairs = _with_near_duplicates(n_base=600, n_dups=200, dim=32)
        train, test = deduplicated_split(X, train_size=0.8)
        assert _valid(train, test, len(X))
        in_train = np.isin(np.arange(len(X)), train)
        assert np.all(in_train[pairs[:, 0]] == in_train[pairs[:, 1]])
        assert abs(len(train) / len(X) - 0.8) < 0.05

    def test_warns_when_groups_skew_the_split(self):
        """Indivisible groups of 50 and 40 plus 10 singletons cannot reach a
        70-sample train set (the closest is 60); that must be reported."""
        rng = np.random.RandomState(0)
        X = np.vstack([
            rng.randn(50, 4) * 1e-3,                  # tight group of 50
            rng.randn(40, 4) * 1e-3 + 1000,           # tight group of 40
            rng.randn(10, 4) * 100 + 5000,            # 10 isolated points
        ])
        with pytest.warns(UserWarning, match="largest with 50 of 100"):
            deduplicated_split(X, train_size=0.7, similarity_threshold=0.1)


def _with_near_duplicates(n_base, n_dups, dim, noise=1e-3, seed=0):
    """Random points plus ``n_dups`` noisy copies; returns (X, pairs) where each
    row of ``pairs`` is (original, copy)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_base, dim))
    src = rng.choice(n_base, n_dups, replace=False)
    X = np.vstack([base, base[src] + noise * rng.normal(size=(n_dups, dim))])
    return X, np.column_stack([src, n_base + np.arange(n_dups)])


def _same_partition(a, b):
    """Whether two label arrays describe the same partition up to relabeling."""
    return len(set(zip(a.tolist(), b.tolist(), strict=True))) == len(set(a)) == len(set(b))


class TestNearDuplicateComponents:

    @staticmethod
    def _reference(X, threshold, metric):
        """The original all-pairs construction, kept as an exact oracle."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial.distance import cdist
        d = cdist(X, X, metric=metric)
        np.fill_diagonal(d, np.inf)
        return connected_components(csr_matrix(d <= threshold), directed=False)[1]

    @pytest.mark.parametrize(
        "metric", ["euclidean", "sqeuclidean", "cosine", "cityblock", "jensenshannon"]
    )
    def test_matches_all_pairs_reference(self, metric):
        """Many small components, explicit threshold: same partition as the
        full-matrix construction, on both the sklearn and cdist paths."""
        from scipy.spatial.distance import pdist
        X = np.abs(np.random.RandomState(0).rand(400, 3))
        threshold = float(np.percentile(pdist(X, metric=metric), 0.2))
        labels, used = near_duplicate_components(X, threshold, metric)
        assert used == threshold
        ref = self._reference(X, threshold, metric)
        assert 1 < len(set(ref)) < len(X)  # non-trivial partition
        assert _same_partition(labels, ref)

    def test_chunked_cdist_path_matches(self, monkeypatch):
        import splytters.utils as utils
        X = np.abs(np.random.RandomState(1).rand(200, 4))
        expected = near_duplicate_components(X, None, "jensenshannon")
        monkeypatch.setattr(utils, "_CDIST_CHUNK_ELEMENTS", 5)  # one row per chunk
        labels, threshold = near_duplicate_components(X, None, "jensenshannon")
        assert threshold == pytest.approx(expected[1])
        assert _same_partition(labels, expected[0])

    @pytest.mark.parametrize("metric", ["euclidean", "cosine"])
    def test_exact_duplicates_linked_at_zero_threshold(self, metric):
        """sklearn's dot-product distances put exact duplicates up to ~1e-5
        apart at this scale; they must still count as distance 0."""
        base = np.random.default_rng(0).normal(size=(300, 512)) * 10
        X = np.vstack([base, base[:100]])
        labels, _ = near_duplicate_components(X, 0.0, metric)
        assert np.array_equal(labels[:100], labels[300:])
        assert len(set(labels)) == 300

    def test_default_links_nothing_without_duplicates(self):
        X = np.random.default_rng(0).normal(size=(2000, 64))
        labels, _ = near_duplicate_components(X, None, "euclidean")
        assert len(set(labels)) == len(X)

    @pytest.mark.parametrize("frac", [0.1, 0.6])
    def test_default_finds_injected_duplicates(self, frac):
        """Including when most samples are duplicated, where a 1st-neighbor
        scale would collapse to the duplicates' own spacing."""
        n_base = 1000
        X, pairs = _with_near_duplicates(n_base, int(frac * n_base), dim=32)
        labels, _ = near_duplicate_components(X, None, "euclidean")
        assert np.all(labels[pairs[:, 0]] == labels[pairs[:, 1]])
        assert np.bincount(labels).max() == 2  # no accidental merging
