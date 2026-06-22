"""Tests for diversity metrics (within-set spread)."""

import numpy as np
import pytest

from splytters.metrics import diversity_text, mean_dist


class TestMeanDist:

    def test_identical_points_have_zero_spread(self):
        X = np.ones((10, 3))
        assert mean_dist(X) == pytest.approx(0.0)

    def test_matches_manual_euclidean(self):
        rng = np.random.RandomState(0)
        X = rng.randn(50, 4)
        centroid = X.mean(axis=0)
        expected = np.mean([np.linalg.norm(row - centroid) for row in X])
        assert mean_dist(X) == pytest.approx(expected)

    def test_more_spread_is_larger(self):
        rng = np.random.RandomState(1)
        tight = rng.randn(40, 2) * 0.1
        wide = rng.randn(40, 2) * 5.0
        assert mean_dist(wide) > mean_dist(tight)

    def test_custom_distance_callable(self):
        X = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])

        # Manhattan distance to centroid [2, 0]: |x-2| summed -> mean of 2,0,2.
        def manhattan(u, v):
            return float(np.abs(np.asarray(u) - np.asarray(v)).sum())

        assert mean_dist(X, distance=manhattan) == pytest.approx((2 + 0 + 2) / 3)

    def test_empty_returns_zero(self):
        assert mean_dist(np.empty((0, 3))) == 0.0

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            mean_dist(np.array([1.0, 2.0, 3.0]))


class TestDiversityText:

    def test_fewer_than_two_is_zero(self):
        assert diversity_text([]) == 0.0
        assert diversity_text(["only one"]) == 0.0

    def test_identical_corpus_has_zero_diversity(self):
        assert diversity_text(["a b c", "a b c", "a b c"]) == pytest.approx(0.0)

    def test_varied_corpus_is_positive(self):
        d = diversity_text(["the cat sat", "a dog ran", "blue green sky"])
        assert d > 0.0

    def test_subsampling_runs_and_is_bounded(self):
        corpus = [f"word{i} token here" for i in range(30)]
        d = diversity_text(corpus, sample_size=8, random_state=0)
        assert 0.0 <= d <= 1.0
