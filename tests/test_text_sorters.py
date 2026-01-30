"""Unit tests for text_sorters.py"""

import pytest

from sorters.text_sorters import (
    simple_tokenizer,
    character_length,
    tokens_length,
    sentence_count,
    lexical_diversity,
    vocabulary_rarity,
    perplexity_score,
    readability_score,
    _score_sorter,
)


class TestSimpleTokenizer:
    """Tests for simple_tokenizer function."""

    def test_splits_on_whitespace(self):
        """Should split text on whitespace."""
        result = simple_tokenizer("hello world")
        assert result == ["hello", "world"]

    def test_multiple_spaces(self):
        """Should handle multiple spaces."""
        result = simple_tokenizer("hello   world")
        assert result == ["hello", "world"]

    def test_empty_string(self):
        """Should return empty list for empty string."""
        result = simple_tokenizer("")
        assert result == []

    def test_single_word(self):
        """Should handle single word."""
        result = simple_tokenizer("hello")
        assert result == ["hello"]


class TestCharacterLength:
    """Tests for character_length function."""

    @pytest.fixture
    def texts(self):
        return ["hi", "hello", "greetings"]  # 2, 5, 9 chars

    def test_orders_short_first(self, texts):
        """Shortest texts should come first when short_first=True."""
        result = character_length(texts, short_first=True)
        assert result == ["hi", "hello", "greetings"]

    def test_orders_long_first(self, texts):
        """Longest texts should come first when short_first=False."""
        result = character_length(texts, short_first=False)
        assert result == ["greetings", "hello", "hi"]

    def test_preserves_original(self, texts):
        """Original list should not be modified."""
        original = texts.copy()
        character_length(texts, short_first=True)
        assert texts == original


class TestTokensLength:
    """Tests for tokens_length function."""

    @pytest.fixture
    def texts(self):
        return [
            "one",                    # 1 token
            "one two",                # 2 tokens
            "one two three four",    # 4 tokens
        ]

    def test_orders_short_first(self, texts):
        """Fewest tokens should come first when short_first=True."""
        result = tokens_length(texts, short_first=True)
        assert result == [["one"], ["one", "two"], ["one", "two", "three", "four"]]

    def test_orders_long_first(self, texts):
        """Most tokens should come first when short_first=False."""
        result = tokens_length(texts, short_first=False)
        assert result == [["one", "two", "three", "four"], ["one", "two"], ["one"]]

    def test_custom_tokenizer(self):
        """Should use custom tokenizer when provided."""
        texts = ["a,b,c", "a,b"]
        result = tokens_length(texts, tokenizer=lambda s: s.split(","))
        assert result == [["a", "b"], ["a", "b", "c"]]


class TestSentenceCount:
    """Tests for sentence_count function."""

    @pytest.fixture
    def texts(self):
        return [
            "Hello.",                                    # 1 sentence
            "Hello. How are you?",                       # 2 sentences
            "Hello. How are you? I am fine. Thanks!",   # 4 sentences
        ]

    def test_orders_few_sentences_first(self, texts):
        """Fewer sentences should come first when low_first=True."""
        result = sentence_count(texts, low_first=True)
        indices = [idx for idx, _ in result]
        assert indices == [0, 1, 2]

    def test_orders_many_sentences_first(self, texts):
        """More sentences should come first when low_first=False."""
        result = sentence_count(texts, low_first=False)
        indices = [idx for idx, _ in result]
        assert indices == [2, 1, 0]

    def test_returns_correct_counts(self, texts):
        """Should return correct sentence counts."""
        result = sentence_count(texts, low_first=True)
        counts = {idx: count for idx, count in result}
        assert counts[0] == 1
        assert counts[1] == 2
        assert counts[2] == 4

    def test_returns_index_count_tuples(self, texts):
        """Results should be (index, count) tuples."""
        result = sentence_count(texts)
        for item in result:
            assert len(item) == 2
            assert isinstance(item[0], int)
            assert isinstance(item[1], int)


class TestLexicalDiversity:
    """Tests for lexical_diversity function."""

    @pytest.fixture
    def texts(self):
        return [
            "the the the the",           # TTR = 1/4 = 0.25 (repetitive)
            "the cat sat mat",            # TTR = 4/4 = 1.0 (diverse)
            "the cat the cat",            # TTR = 2/4 = 0.5 (medium)
        ]

    def test_orders_repetitive_first(self, texts):
        """Repetitive texts (low TTR) should come first when low_first=True."""
        result = lexical_diversity(texts, low_first=True)
        indices = [idx for idx, _ in result]
        assert indices == [0, 2, 1]  # 0.25, 0.5, 1.0

    def test_orders_diverse_first(self, texts):
        """Diverse texts (high TTR) should come first when low_first=False."""
        result = lexical_diversity(texts, low_first=False)
        indices = [idx for idx, _ in result]
        assert indices == [1, 2, 0]  # 1.0, 0.5, 0.25

    def test_ttr_values(self, texts):
        """Should calculate correct TTR values."""
        result = lexical_diversity(texts, low_first=True)
        scores = {idx: ttr for idx, ttr in result}
        assert scores[0] == pytest.approx(0.25)
        assert scores[1] == pytest.approx(1.0)
        assert scores[2] == pytest.approx(0.5)

    def test_empty_text_gets_zero(self):
        """Empty text should receive TTR of 0."""
        result = lexical_diversity(["", "hello world"])
        scores = {idx: ttr for idx, ttr in result}
        assert scores[0] == 0.0


class TestVocabularyRarity:
    """Tests for vocabulary_rarity function."""

    @pytest.fixture
    def texts(self):
        return [
            "the a is",                      # Common words
            "photosynthesis mitochondria",   # Rare/technical words
            "the cat sat",                   # Mix
        ]

    def test_orders_common_first(self, texts):
        """Common vocabulary should come first when low_first=True."""
        result = vocabulary_rarity(texts, low_first=True)
        indices = [idx for idx, _ in result]
        # "the a is" should be first (most common words)
        assert indices[0] == 0

    def test_orders_rare_first(self, texts):
        """Rare vocabulary should come first when low_first=False."""
        result = vocabulary_rarity(texts, low_first=False)
        indices = [idx for idx, _ in result]
        # "photosynthesis mitochondria" should be first (rarest words)
        assert indices[0] == 1

    def test_common_words_have_lower_rarity(self, texts):
        """Common words should have lower rarity scores."""
        result = vocabulary_rarity(texts)
        scores = {idx: rarity for idx, rarity in result}
        # "the a is" should have lower rarity than technical words
        assert scores[0] < scores[1]

    def test_empty_text_gets_zero(self):
        """Empty text should receive rarity of 0."""
        result = vocabulary_rarity(["", "hello"])
        scores = {idx: rarity for idx, rarity in result}
        assert scores[0] == 0.0

    def test_returns_index_rarity_tuples(self, texts):
        """Results should be (index, rarity) tuples."""
        result = vocabulary_rarity(texts)
        for item in result:
            assert len(item) == 2
            assert isinstance(item[0], int)
            assert isinstance(item[1], float)


class TestPerplexityScore:
    """Tests for perplexity_score function."""

    @pytest.fixture
    def texts(self):
        return [
            "The cat sat on the mat.",           # Normal sentence
            "Colorless green ideas sleep furiously.",  # Grammatical but nonsensical
            "asdf jkl qwerty zxcv",              # Random words
        ]

    @pytest.mark.slow
    def test_returns_correct_structure(self, texts):
        """Results should be (index, perplexity) tuples."""
        result = perplexity_score(texts)
        assert len(result) == 3
        for item in result:
            assert len(item) == 2
            assert isinstance(item[0], int)
            assert isinstance(item[1], float)

    @pytest.mark.slow
    def test_normal_text_lower_perplexity(self, texts):
        """Normal text should have lower perplexity than nonsense."""
        result = perplexity_score(texts, low_first=True)
        scores = {idx: ppl for idx, ppl in result}
        # Normal sentence should have lower perplexity than random words
        assert scores[0] < scores[2]

    @pytest.mark.slow
    def test_short_text_gets_infinity(self):
        """Text too short for perplexity should get infinity."""
        result = perplexity_score(["a", "The cat sat on the mat."])
        scores = {idx: ppl for idx, ppl in result}
        assert scores[0] == float("inf")

    @pytest.mark.slow
    def test_orders_typical_first(self, texts):
        """Typical text should come first when low_first=True."""
        result = perplexity_score(texts, low_first=True)
        indices = [idx for idx, _ in result]
        # Normal sentence should be first
        assert indices[0] == 0


class TestReadabilityScore:
    """Tests for readability_score function."""

    @pytest.fixture
    def texts(self):
        return [
            # Simple text (low grade level)
            "The cat sat. The dog ran. It was fun.",
            # Complex text (high grade level)
            "The implementation of sophisticated algorithmic procedures necessitates comprehensive understanding of computational complexity theory and abstract mathematical concepts.",
        ]

    def test_orders_simple_first(self, texts):
        """Simple text should come first when low_first=True."""
        result = readability_score(texts, metric="flesch_kincaid", low_first=True)
        indices = [idx for idx, _ in result]
        assert indices[0] == 0  # Simple text first

    def test_orders_complex_first(self, texts):
        """Complex text should come first when low_first=False."""
        result = readability_score(texts, metric="flesch_kincaid", low_first=False)
        indices = [idx for idx, _ in result]
        assert indices[0] == 1  # Complex text first

    def test_simple_has_lower_score(self, texts):
        """Simple text should have lower Flesch-Kincaid score."""
        result = readability_score(texts, metric="flesch_kincaid")
        scores = {idx: score for idx, score in result}
        assert scores[0] < scores[1]

    def test_different_metrics(self, texts):
        """Should support different readability metrics."""
        metrics = ["flesch_kincaid", "flesch", "gunning_fog", "coleman_liau", "ari"]
        for metric in metrics:
            result = readability_score(texts, metric=metric)
            assert len(result) == 2

    def test_invalid_metric_raises(self, texts):
        """Should raise ValueError for invalid metric."""
        with pytest.raises(ValueError, match="Unknown readability metric"):
            readability_score(texts, metric="invalid_metric")

    def test_short_text_gets_infinity(self):
        """Text too short for readability should get infinity."""
        result = readability_score(["Hi.", "The cat sat on the comfortable mat in the house."])
        scores = {idx: score for idx, score in result}
        assert scores[0] == float("inf")

    def test_returns_index_score_tuples(self, texts):
        """Results should be (index, score) tuples."""
        result = readability_score(texts)
        for item in result:
            assert len(item) == 2
            assert isinstance(item[0], int)
            assert isinstance(item[1], float)


class TestScoreSorter:
    """Tests for _score_sorter function."""

    def test_sorts_by_score_ascending(self):
        """Should sort by score ascending when low_first=True."""
        data = [3, 1, 2]
        result = _score_sorter(data, lambda x: x, low_first=True)
        assert result == [1, 2, 3]

    def test_sorts_by_score_descending(self):
        """Should sort by score descending when low_first=False."""
        data = [3, 1, 2]
        result = _score_sorter(data, lambda x: x, low_first=False)
        assert result == [3, 2, 1]

    def test_custom_score_function(self):
        """Should use custom score function."""
        data = ["aaa", "b", "cc"]
        result = _score_sorter(data, len, low_first=True)
        assert result == ["b", "cc", "aaa"]

    def test_preserves_original(self):
        """Original data should not be modified."""
        data = [3, 1, 2]
        original = data.copy()
        _score_sorter(data, lambda x: x)
        assert data == original

    def test_handles_empty_list(self):
        """Should handle empty list."""
        result = _score_sorter([], lambda x: x)
        assert result == []


class TestEdgeCases:
    """Test edge cases for all text sorters."""

    def test_character_length_empty_list(self):
        """character_length should handle empty list."""
        assert character_length([]) == []

    def test_character_length_single_item(self):
        """character_length should handle single item."""
        result = character_length(["hello"])
        assert result == ["hello"]

    def test_tokens_length_empty_list(self):
        """tokens_length should handle empty list."""
        assert tokens_length([]) == []

    def test_tokens_length_single_item(self):
        """tokens_length should handle single item."""
        result = tokens_length(["hello world"])
        assert result == [["hello", "world"]]

    def test_sentence_count_empty_list(self):
        """sentence_count should handle empty list."""
        assert sentence_count([]) == []

    def test_sentence_count_single_item(self):
        """sentence_count should handle single item."""
        result = sentence_count(["Hello."])
        assert len(result) == 1
        assert result[0][0] == 0

    def test_lexical_diversity_empty_list(self):
        """lexical_diversity should handle empty list."""
        assert lexical_diversity([]) == []

    def test_lexical_diversity_single_item(self):
        """lexical_diversity should handle single item."""
        result = lexical_diversity(["hello world"])
        assert len(result) == 1
        assert result[0][0] == 0

    def test_vocabulary_rarity_empty_list(self):
        """vocabulary_rarity should handle empty list."""
        assert vocabulary_rarity([]) == []

    def test_vocabulary_rarity_single_item(self):
        """vocabulary_rarity should handle single item."""
        result = vocabulary_rarity(["hello"])
        assert len(result) == 1
        assert result[0][0] == 0

    def test_readability_score_empty_list(self):
        """readability_score should handle empty list."""
        assert readability_score([]) == []

    def test_readability_score_single_item(self):
        """readability_score should handle single item."""
        result = readability_score(["The cat sat on the mat in the house."])
        assert len(result) == 1
        assert result[0][0] == 0

    @pytest.mark.slow
    def test_perplexity_score_empty_list(self):
        """perplexity_score should handle empty list."""
        assert perplexity_score([]) == []

    @pytest.mark.slow
    def test_perplexity_score_single_item(self):
        """perplexity_score should handle single item."""
        result = perplexity_score(["The cat sat on the mat."])
        assert len(result) == 1
        assert result[0][0] == 0
