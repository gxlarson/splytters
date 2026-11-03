"""Tests for the newly added sorters.

mahalanobis_distance_to_mean, knn_label_disagreement (embedding),
gzip_complexity (text), and sharpness (image).
"""

import numpy as np
import pytest

from splytters.sorters import (
    gzip_complexity,
    knn_label_disagreement,
    mahalanobis_distance_to_mean,
)


def _is_full_ranking(result, n):
    return sorted(i for i, _ in result) == list(range(n))


class TestMahalanobisDistanceToMean:

    def test_full_ranking_ascending(self):
        X = np.random.RandomState(0).randn(50, 6)
        r = mahalanobis_distance_to_mean(X)
        assert _is_full_ranking(r, 50)
        assert all(r[i][1] <= r[i + 1][1] for i in range(len(r) - 1))

    def test_outlier_ranks_last(self):
        X = np.random.RandomState(0).randn(40, 4)
        X[0] = 50.0  # extreme outlier
        r = mahalanobis_distance_to_mean(X, low_first=True)
        assert r[-1][0] == 0

    def test_low_first_reverses(self):
        X = np.random.RandomState(1).randn(30, 5)
        a = mahalanobis_distance_to_mean(X, low_first=True)
        b = mahalanobis_distance_to_mean(X, low_first=False)
        assert a[0][0] == b[-1][0]


class TestKnnLabelDisagreement:

    def test_boundary_point_scores_high(self):
        rng = np.random.RandomState(0)
        X = np.vstack([rng.randn(30, 2) + [5, 5], rng.randn(30, 2) + [-5, -5]])
        y = np.array([0] * 30 + [1] * 30)
        X[0] = [-5, -5]  # a class-0 point planted in class-1 territory
        scores = dict(knn_label_disagreement(X, y, k=5))
        assert scores[0] > 0.5            # surrounded by the other class
        assert min(scores.values()) == 0.0  # interior points fully agree

    def test_range_and_full_ranking(self):
        rng = np.random.RandomState(0)
        X = rng.randn(40, 4)
        y = rng.randint(0, 3, 40)
        r = knn_label_disagreement(X, y, k=4)
        assert _is_full_ranking(r, 40)
        assert all(0.0 <= v <= 1.0 for _, v in r)

    def test_duplicate_ties_remove_self_by_index(self):
        X = np.zeros((6, 2))
        y = np.array([0, 1, 1, 1, 1, 1])
        scores = dict(knn_label_disagreement(X, y, k=1))
        assert scores[0] == 1.0

    def test_k_must_be_positive(self):
        X = np.arange(12, dtype=float).reshape(6, 2)
        y = np.array([0, 0, 0, 1, 1, 1])
        with pytest.raises(ValueError, match="k must be a positive integer"):
            knn_label_disagreement(X, y, k=0)


class TestGzipComplexity:

    def test_redundant_more_compressible_than_diverse(self):
        redundant = "the cat sat on the mat. " * 40
        diverse = "Quantum entanglement perplexes physicists worldwide. " * 5
        r = dict(gzip_complexity([redundant, diverse], low_first=True))
        assert r[0] < r[1]  # repetitive text compresses better (lower ratio)

    def test_full_ranking(self):
        r = gzip_complexity(["a" * 200, "abcd" * 50, "hello world " * 20])
        assert _is_full_ranking(r, 3)

    def test_empty_string_is_zero(self):
        r = dict(gzip_complexity(["", "aaaa" * 100]))
        assert r[0] == 0.0


class TestSharpness:

    def test_sharp_ranks_above_blurry(self):
        Image = pytest.importorskip("PIL.Image")
        from splytters.sorters import sharpness

        rng = np.random.RandomState(0)
        blur = Image.fromarray(np.full((64, 64), 128, dtype="uint8"))
        sharp = Image.fromarray((rng.rand(64, 64) * 255).astype("uint8"))
        scores = dict(sharpness([blur, sharp], low_first=True))
        assert scores[1] > scores[0]  # noisy image has higher Laplacian variance
