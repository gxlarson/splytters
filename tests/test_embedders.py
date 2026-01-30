"""Unit tests for text embedders using real test data."""

import os

import numpy as np
import pytest

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "test_data", "text", "bank_balance_queries.txt")


@pytest.fixture(scope="module")
def queries():
    with open(DATA_PATH) as f:
        return [line.strip() for line in f if line.strip()]


@pytest.fixture(scope="module")
def balance_queries(queries):
    """First 30 queries that contain 'balance' and 'what'."""
    return queries[:30]


@pytest.fixture(scope="module")
def alt_queries(queries):
    """Last 20 queries that avoid 'bank', 'balance', 'what'."""
    return queries[30:]


# ---------------------------------------------------------------------------
# TextEmbedder (SentenceTransformer)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def text_embedder():
    from embedders import TextEmbedder
    return TextEmbedder()


class TestTextEmbedder:

    def test_output_shape(self, text_embedder, queries):
        embs = text_embedder.embed(queries)
        assert isinstance(embs, np.ndarray)
        assert embs.shape[0] == 50
        assert embs.ndim == 2

    def test_no_nan_values(self, text_embedder, queries):
        embs = text_embedder.embed(queries)
        assert not np.isnan(embs).any()

    def test_similar_queries_closer_than_dissimilar(self, text_embedder, balance_queries, alt_queries):
        """Two balance-keyword queries should be closer to each other than to an alt query."""
        b_embs = text_embedder.embed(balance_queries[:2])
        a_embs = text_embedder.embed(alt_queries[:1])
        within_dist = np.linalg.norm(b_embs[0] - b_embs[1])
        cross_dist = np.linalg.norm(b_embs[0] - a_embs[0])
        # Not guaranteed for every pair, but these two are near-identical phrasings
        assert within_dist < cross_dist

    def test_single_query(self, text_embedder):
        embs = text_embedder.embed(["how much money do I have"])
        assert embs.shape[0] == 1


# ---------------------------------------------------------------------------
# CLIPTextEmbedder
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clip_text_embedder():
    from embedders import CLIPTextEmbedder
    return CLIPTextEmbedder()


class TestCLIPTextEmbedder:

    def test_output_shape(self, clip_text_embedder, queries):
        embs = clip_text_embedder.embed(queries)
        assert isinstance(embs, np.ndarray)
        assert embs.shape[0] == 50
        assert embs.ndim == 2

    def test_no_nan_values(self, clip_text_embedder, queries):
        embs = clip_text_embedder.embed(queries)
        assert not np.isnan(embs).any()

    def test_single_query(self, clip_text_embedder):
        embs = clip_text_embedder.embed(["show me my account total"])
        assert embs.shape[0] == 1
