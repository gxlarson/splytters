"""Unit tests for per-class application of splitters (splytters.stratify)."""

import numpy as np
import pytest

from splytters.adversarial import cluster_split, distance_adversarial_split
from splytters.curriculum import sorted_stratified_split
from splytters.sorters.embedding_sorters import distance_to_mean
from splytters.sorters.text_sorters import character_length
from splytters.stratify import per_class_sort, per_class_split
from splytters.utils import random_split


@pytest.fixture
def labeled_embeddings():
    """120 points in 4 classes of 30, each its own well-separated 8-D blob."""
    rng = np.random.RandomState(0)
    X, y = [], []
    for c in range(4):
        center = np.zeros(8)
        center[c] = 10.0
        X.append(rng.randn(30, 8) + center)
        y.extend([c] * 30)
    return np.vstack(X), np.array(y)


def _is_partition(train, test, n):
    """train and test are disjoint and together cover range(n) exactly."""
    both = np.concatenate([train, test])
    return len(both) == n and set(both.tolist()) == set(range(n))


class TestPerClassSplit:
    def test_partition(self, labeled_embeddings):
        X, y = labeled_embeddings
        train, test = per_class_split(distance_adversarial_split, X, y, 0.7)
        assert _is_partition(train, test, len(X))

    def test_full_train_coverage(self, labeled_embeddings):
        """Every class appears in train AND test — the whole point."""
        X, y = labeled_embeddings
        train, test = per_class_split(distance_adversarial_split, X, y, 0.7)
        assert set(y[train].tolist()) == set(np.unique(y).tolist())
        assert set(y[test].tolist()) == set(np.unique(y).tolist())

    def test_per_class_proportion(self, labeled_embeddings):
        """train_size is applied within each class, not globally."""
        X, y = labeled_embeddings
        train, _ = per_class_split(distance_adversarial_split, X, y, 0.7)
        for c in np.unique(y):
            n_c = int((y == c).sum())
            assert (y[train] == c).sum() == int(n_c * 0.7)

    def test_sorted_output(self, labeled_embeddings):
        X, y = labeled_embeddings
        train, test = per_class_split(distance_adversarial_split, X, y, 0.7)
        assert np.all(np.diff(train) > 0)
        assert np.all(np.diff(test) > 0)

    def test_deterministic(self, labeled_embeddings):
        X, y = labeled_embeddings
        a = per_class_split(distance_adversarial_split, X, y, 0.7)
        b = per_class_split(distance_adversarial_split, X, y, 0.7)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_random_state_drives_wrapped_splitter(self, labeled_embeddings):
        """random_state must reach split_fn (not just the fallback): a
        seed-varying splitter gives different splits for different seeds, and the
        same seed is reproducible."""
        X, y = labeled_embeddings
        a = per_class_split(cluster_split, X, y, 0.7, random_state=0, n_clusters=5)
        b = per_class_split(cluster_split, X, y, 0.7, random_state=1, n_clusters=5)
        c = per_class_split(cluster_split, X, y, 0.7, random_state=0, n_clusters=5)
        assert not np.array_equal(a[1], b[1])          # seed changes the split
        np.testing.assert_array_equal(a[1], c[1])       # same seed reproducible

    def test_custom_split_fn_without_random_state(self, labeled_embeddings):
        """A split_fn that doesn't accept random_state is called without it."""
        X, y = labeled_embeddings

        def custom(emb, train_size=0.7):
            k = int(len(emb) * train_size)
            return np.arange(k), np.arange(k, len(emb))

        train, test = per_class_split(custom, X, y, 0.7, random_state=0)
        assert _is_partition(train, test, len(X))

    def test_fallback_on_small_class(self):
        """cluster_split(n_clusters=10) fails on a 4-sample class; fallback saves it."""
        rng = np.random.RandomState(1)
        X = np.vstack([rng.randn(40, 6), rng.randn(4, 6) + 20])
        y = np.array([0] * 40 + [1] * 4)
        # on_error='raise' surfaces the KMeans failure (n_clusters > n_samples)...
        with pytest.raises(ValueError):
            per_class_split(cluster_split, X, y, 0.7, on_error="raise", n_clusters=10)
        # ...while the default fallback keeps both classes covered.
        train, test = per_class_split(cluster_split, X, y, 0.7, n_clusters=10)
        assert _is_partition(train, test, len(X))
        assert {0, 1} <= set(y[train].tolist())
        assert {0, 1} <= set(y[test].tolist())

    def test_singleton_class_goes_to_train(self):
        rng = np.random.RandomState(2)
        X = np.vstack([rng.randn(10, 4), rng.randn(1, 4) + 9])
        y = np.array([0] * 10 + [1])
        train, test = per_class_split(random_split, X, y, 0.7)
        assert 10 in train          # the lone class-1 sample
        assert 10 not in test
        assert _is_partition(train, test, len(X))

    def test_length_mismatch_raises(self, labeled_embeddings):
        X, y = labeled_embeddings
        with pytest.raises(ValueError, match="rows but"):
            per_class_split(random_split, X, y[:-1], 0.7)

    def test_invalid_on_error_raises(self, labeled_embeddings):
        X, y = labeled_embeddings
        with pytest.raises(ValueError, match="on_error"):
            per_class_split(random_split, X, y, 0.7, on_error="nope")


class TestPerClassSort:
    def test_is_permutation(self, labeled_embeddings):
        X, y = labeled_embeddings
        order = per_class_sort(distance_to_mean, X, y)
        idx = [i for i, _ in order]
        assert sorted(idx) == list(range(len(X)))

    def test_ranks_within_class_centroid(self, labeled_embeddings):
        """Within a class, samples are ordered by distance to THAT class's centroid."""
        X, y = labeled_embeddings
        order = per_class_sort(distance_to_mean, X, y)
        for c in np.unique(y):
            global_seq = [i for i, _ in order if y[i] == c]
            class_idx = np.flatnonzero(y == c)
            expected = [int(class_idx[li]) for li, _ in distance_to_mean(X[class_idx])]
            assert global_seq == expected

    def test_differs_from_global(self, labeled_embeddings):
        """Per-class ranking is genuinely different from the global one."""
        X, y = labeled_embeddings
        per_class = [i for i, _ in per_class_sort(distance_to_mean, X, y)]
        global_order = [i for i, _ in distance_to_mean(X)]
        assert per_class != global_order

    def test_feeds_sorted_stratified_split(self, labeled_embeddings):
        """The combined ranking drops straight into sorted_stratified_split."""
        X, y = labeled_embeddings
        order = per_class_sort(distance_to_mean, X, y)
        train, test = sorted_stratified_split(order, y, 0.7)
        assert _is_partition(train, test, len(X))
        assert set(y[train].tolist()) == set(np.unique(y).tolist())  # coverage 1.0

    def test_text_sorter_on_list_data(self):
        """Works with list-of-strings data, not just arrays."""
        texts = ["a", "abcd", "ab", "abcdef", "xyz", "x"]
        y = np.array([0, 0, 0, 1, 1, 1])
        order = per_class_sort(character_length, texts, y)
        assert sorted(i for i, _ in order) == list(range(len(texts)))
        # character_length is intrinsic, so within-class order is by raw length.
        c0 = [i for i, _ in order if y[i] == 0]
        assert c0 == [0, 2, 1]  # "a"(1) < "ab"(2) < "abcd"(4)

    def test_length_mismatch_raises(self, labeled_embeddings):
        X, y = labeled_embeddings
        with pytest.raises(ValueError, match="items but"):
            per_class_sort(distance_to_mean, X, y[:-1])

    def test_fallback_on_error(self, labeled_embeddings):
        X, y = labeled_embeddings

        def boom(_data):
            raise RuntimeError("sorter failed")

        # fallback keeps every sample (in original order); raise propagates.
        order = per_class_sort(boom, X, y, on_error="fallback")
        assert sorted(i for i, _ in order) == list(range(len(X)))
        with pytest.raises(RuntimeError):
            per_class_sort(boom, X, y, on_error="raise")
