"""Tests for split-quality reporting."""

import numpy as np
import pytest

from splytters import (
    centroid_adversarial_split,
    cluster_leak_split,
    cluster_split,
    compare_splitters,
    random_split,
    split_report,
)


@pytest.fixture
def embeddings_2d():
    rng = np.random.RandomState(42)
    a = rng.randn(60, 2) + np.array([5, 5])
    b = rng.randn(60, 2) + np.array([-5, -5])
    return np.vstack([a, b])


class TestSplitReport:

    def test_has_expected_keys(self, embeddings_2d):
        train, test = random_split(embeddings_2d)
        rep = split_report(embeddings_2d, train, test)
        for key in (
            "n_train", "n_test", "train_fraction",
            "centroid_distance", "mean_cross_distance", "coverage",
            "cluster_leakage_ratio",
            "mmd_rbf", "energy_distance", "wasserstein_mean", "ks_mean",
        ):
            assert key in rep, f"missing {key}"
            assert np.isfinite(rep[key])

    def test_counts_consistent(self, embeddings_2d):
        train, test = random_split(embeddings_2d, train_size=0.7)
        rep = split_report(embeddings_2d, train, test)
        assert rep["n_train"] == len(train)
        assert rep["n_test"] == len(test)
        assert rep["train_fraction"] == pytest.approx(len(train) / len(embeddings_2d))

    def test_adversarial_more_distant_than_random(self, embeddings_2d):
        """An adversarial split should separate train/test more than random."""
        r_tr, r_te = random_split(embeddings_2d, random_state=0)
        a_tr, a_te = centroid_adversarial_split(embeddings_2d, n_clusters=4)
        rand = split_report(embeddings_2d, r_tr, r_te)
        adv = split_report(embeddings_2d, a_tr, a_te)
        # Adversarial pushes test away from train on multiple measures.
        assert adv["mean_cross_distance"] >= rand["mean_cross_distance"]
        assert adv["energy_distance"] >= rand["energy_distance"]

    def test_label_shift_reported(self, embeddings_2d):
        y = np.array([0] * 60 + [1] * 60)
        train, test = random_split(embeddings_2d)
        rep = split_report(embeddings_2d, train, test, y=y)
        assert "label_distribution_shift" in rep
        assert 0.0 <= rep["label_distribution_shift"] <= 1.0

    def test_rejects_nan_embeddings(self, embeddings_2d):
        X = embeddings_2d.copy()
        X[0, 0] = np.nan
        train, test = random_split(embeddings_2d)
        with pytest.raises(ValueError):
            split_report(X, train, test)


class TestCompareSplitters:

    def test_returns_row_per_splitter(self, embeddings_2d):
        table = compare_splitters(
            embeddings_2d,
            {
                "random": random_split,
                "adversarial": cluster_split,
                "overlap": cluster_leak_split,
            },
        )
        assert set(table.keys()) == {"random", "adversarial", "overlap"}
        for row in table.values():
            assert "mmd_rbf" in row
