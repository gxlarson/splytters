"""Tests for the scikit-learn compatibility layer and framework interop."""

import numpy as np
import pytest

from splytters import (
    SplytterSplit,
    adversarial_train_test_split,
    balanced_train_test_split,
    cluster_split,
    distance_adversarial_split,
    overlap_train_test_split,
    splytter_train_test_split,
)
from splytters.interop import split_dataframe, split_dataset, to_torch_subsets


@pytest.fixture
def labelled_data():
    """120 points in two well-separated 2D clusters with binary labels."""
    rng = np.random.RandomState(42)
    a = rng.randn(60, 2) + np.array([5, 5])
    b = rng.randn(60, 2) + np.array([-5, -5])
    X = np.vstack([a, b])
    y = np.array([0] * 60 + [1] * 60)
    return X, y


# ---------------------------------------------------------------------------
# SplytterSplit cross-validator
# ---------------------------------------------------------------------------

class TestSplytterSplit:

    def test_get_n_splits_default_one(self, labelled_data):
        X, _ = labelled_data
        cv = SplytterSplit(embeddings=X)
        assert cv.get_n_splits() == 1

    def test_split_is_a_partition(self, labelled_data):
        X, _ = labelled_data
        cv = SplytterSplit(cluster_split, embeddings=X, n_clusters=4)
        (train, test), = list(cv.split(X))
        assert isinstance(train, np.ndarray) and isinstance(test, np.ndarray)
        assert set(train.tolist()) | set(test.tolist()) == set(range(len(X)))
        assert set(train.tolist()) & set(test.tolist()) == set()

    def test_n_splits_repeats_differ(self, labelled_data):
        X, _ = labelled_data
        cv = SplytterSplit(
            distance_adversarial_split, embeddings=X, n_splits=1
        )
        assert len(list(cv.split(X))) == 1

    def test_uses_X_when_embeddings_none(self, labelled_data):
        X, _ = labelled_data
        cv = SplytterSplit(cluster_split, n_clusters=4)
        folds = list(cv.split(X))
        assert len(folds) == 1

    def test_custom_splitter_without_random_state(self, labelled_data):
        """SplytterSplit must not force random_state onto a splitter that can't
        accept it (previously TypeError on the first .split() call)."""
        X, _ = labelled_data

        def head_tail_split(embeddings, train_size=0.7):
            n = len(embeddings)
            n_train = int(n * train_size)
            idx = np.arange(n)
            return idx[:n_train], idx[n_train:]

        cv = SplytterSplit(head_tail_split, embeddings=X)
        (train, test), = list(cv.split(X))
        assert set(train.tolist()) | set(test.tolist()) == set(range(len(X)))

    def test_drop_in_cross_validate(self, labelled_data):
        """The headline guarantee: usable as cv= in cross_validate."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_validate

        X, y = labelled_data
        cv = SplytterSplit(cluster_split, embeddings=X, n_clusters=4)
        results = cross_validate(
            LogisticRegression(), X, y, cv=cv, scoring="accuracy"
        )
        assert "test_score" in results
        assert len(results["test_score"]) == 1


# ---------------------------------------------------------------------------
# train_test_split-style convenience
# ---------------------------------------------------------------------------

class TestTrainTestSplitConvenience:

    def test_indices_only(self, labelled_data):
        X, _ = labelled_data
        train_idx, test_idx = splytter_train_test_split(embeddings=X)
        assert len(train_idx) + len(test_idx) == len(X)

    def test_two_arrays_four_outputs(self, labelled_data):
        X, y = labelled_data
        texts = [f"item-{i}" for i in range(len(X))]
        X_tr, X_te, y_tr, y_te, t_tr, t_te = splytter_train_test_split(
            X, y, texts, embeddings=X, n_clusters=4
        )
        assert len(X_tr) == len(y_tr) == len(t_tr)
        assert len(X_te) == len(y_te) == len(t_te)
        assert len(X_tr) + len(X_te) == len(X)
        # texts is a Python list -> stays a list
        assert isinstance(t_tr, list)

    def test_family_wrappers(self, labelled_data):
        X, y = labelled_data
        for fn in (
            adversarial_train_test_split,
            overlap_train_test_split,
            balanced_train_test_split,
        ):
            out = fn(X, y, embeddings=X)
            assert len(out) == 4

    def test_requires_array_or_embeddings(self):
        with pytest.raises(ValueError, match="at least one array"):
            splytter_train_test_split()

    def test_mismatched_array_length_raises(self, labelled_data):
        """An array whose length differs from embeddings is rejected up front."""
        X, y = labelled_data
        with pytest.raises(ValueError, match="same length"):
            splytter_train_test_split(X, y[:-1], embeddings=X)

    def test_custom_splitter_without_random_state(self, labelled_data):
        """A user splitter that takes neither random_state nor **kwargs must not
        get a random_state forced on it (that raised TypeError before the fix)."""
        X, _ = labelled_data

        def head_tail_split(embeddings, train_size=0.7):
            n = len(embeddings)
            n_train = int(n * train_size)
            idx = np.arange(n)
            return idx[:n_train], idx[n_train:]

        train_idx, test_idx = splytter_train_test_split(
            embeddings=X, splitter=head_tail_split
        )
        assert len(train_idx) + len(test_idx) == len(X)


# ---------------------------------------------------------------------------
# Framework interop (lazily-imported deps)
# ---------------------------------------------------------------------------

class TestPandasInterop:

    def test_split_dataframe_round_trip(self, labelled_data):
        pd = pytest.importorskip("pandas")
        X, y = labelled_data
        df = pd.DataFrame({"a": X[:, 0], "b": X[:, 1], "label": y})
        train_df, test_df = split_dataframe(df, X, n_clusters=4)
        assert len(train_df) + len(test_df) == len(df)
        assert list(train_df.columns) == ["a", "b", "label"]
        # Disjoint original-index labels
        assert set(train_df.index) & set(test_df.index) == set()

    def test_length_mismatch_raises(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"a": range(10)})
        with pytest.raises(ValueError, match="rows but embeddings"):
            split_dataframe(df, np.zeros((9, 2)))


class TestTorchInterop:

    def test_to_torch_subsets(self, labelled_data):
        torch = pytest.importorskip("torch")
        from torch.utils.data import TensorDataset

        X, y = labelled_data
        ds = TensorDataset(torch.tensor(X), torch.tensor(y))
        train_sub, test_sub = to_torch_subsets(ds, X, n_clusters=4)
        assert len(train_sub) + len(test_sub) == len(ds)


class TestHuggingFaceInterop:

    def test_split_dataset(self, labelled_data):
        datasets = pytest.importorskip("datasets")
        X, y = labelled_data
        ds = datasets.Dataset.from_dict(
            {"text": [f"item-{i}" for i in range(len(X))], "label": y.tolist()}
        )
        dd = split_dataset(ds, X, n_clusters=4)
        assert set(dd.keys()) == {"train", "test"}
        assert len(dd["train"]) + len(dd["test"]) == len(ds)
