"""Tests for grouping-aware splits (group_split, deduplicated_split)."""

import numpy as np
import pytest

from splytters import deduplicated_split, group_split


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

    def test_large_group_best_fit_lands_in_train(self):
        """A group larger than the target must not be permanently barred from
        train. With a 10-sample group and a 1-sample group at train_size=0.7,
        best-fit puts the big group in train (realized fraction 10/11 is far
        closer to 0.7 than the 1/11 the old first-fit produced)."""
        X = np.random.RandomState(0).randn(11, 4)
        groups = np.array([0] * 10 + [1])
        train, test = group_split(X, groups, train_size=0.7)
        assert _valid(train, test, len(X))
        assert set(groups[train]) == {0}  # the 10-sample group is in train
        realized = len(train) / len(X)
        assert abs(realized - 0.7) < abs(0.09 - 0.7)


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
