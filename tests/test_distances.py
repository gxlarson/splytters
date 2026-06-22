"""Tests for string-distance primitives used by the diversity metrics."""

import pytest

from splytters.distances import (
    difflib_character_similarity,
    difflib_token_similarity,
    ngram_jaccard_distance,
    ngram_jaccard_similarity,
    simple_tokenizer,
)


def test_simple_tokenizer_splits_on_whitespace():
    assert simple_tokenizer("a b  c") == ["a", "b", "c"]
    assert simple_tokenizer("") == []


class TestNgramJaccard:

    def test_identical_strings_are_maximally_similar(self):
        assert ngram_jaccard_similarity("a b c d", "a b c d") == pytest.approx(1.0)

    def test_disjoint_strings_are_dissimilar(self):
        assert ngram_jaccard_similarity("a a a a", "b b b b") == pytest.approx(0.0)

    def test_symmetric(self):
        s1 = ngram_jaccard_similarity("the quick brown fox", "brown fox the quick")
        s2 = ngram_jaccard_similarity("brown fox the quick", "the quick brown fox")
        assert s1 == pytest.approx(s2)

    def test_similarity_in_unit_interval(self):
        s = ngram_jaccard_similarity("a b c d", "a b x y")
        assert 0.0 <= s <= 1.0

    def test_distance_is_one_minus_similarity(self):
        a, b = "a b c d e", "a b x y z"
        assert ngram_jaccard_distance(a, b) == pytest.approx(
            1.0 - ngram_jaccard_similarity(a, b)
        )

    def test_distance_identical_is_zero(self):
        assert ngram_jaccard_distance("a b c d", "a b c d") == pytest.approx(0.0)


class TestDifflib:

    def test_character_similarity_identical(self):
        assert difflib_character_similarity("hello", "hello") == pytest.approx(1.0)

    def test_character_similarity_disjoint(self):
        assert difflib_character_similarity("aaaa", "bbbb") == pytest.approx(0.0)

    def test_token_similarity_identical(self):
        assert difflib_token_similarity("a b c", "a b c") == pytest.approx(1.0)

    def test_token_similarity_partial(self):
        s = difflib_token_similarity("a b c d", "a b x y")
        assert 0.0 <= s <= 1.0
