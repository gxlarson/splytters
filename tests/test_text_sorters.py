"""Unit tests for text_sorters.py"""

import numpy as np
import pytest

from splytters.sorters.text_sorters import (
    character_length,
    lexical_diversity,
    perplexity_score,
    readability_score,
    sentence_count,
    simple_tokenizer,
    tokens_length,
    vocabulary_rarity,
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
        """Shortest texts should come first when low_first=True."""
        result = character_length(texts, low_first=True)
        indices = [idx for idx, _ in result]
        assert indices == [0, 1, 2]

    def test_orders_long_first(self, texts):
        """Longest texts should come first when low_first=False."""
        result = character_length(texts, low_first=False)
        indices = [idx for idx, _ in result]
        assert indices == [2, 1, 0]

    def test_returns_correct_counts(self, texts):
        """Should return correct character counts."""
        result = character_length(texts, low_first=True)
        scores = {idx: count for idx, count in result}
        assert scores[0] == 2
        assert scores[1] == 5
        assert scores[2] == 9

    def test_returns_index_count_tuples(self, texts):
        """Results should be (index, count) tuples."""
        result = character_length(texts)
        for item in result:
            assert len(item) == 2
            assert isinstance(item[0], int)
            assert isinstance(item[1], int)


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
        """Fewest tokens should come first when low_first=True."""
        result = tokens_length(texts, low_first=True)
        indices = [idx for idx, _ in result]
        assert indices == [0, 1, 2]

    def test_orders_long_first(self, texts):
        """Most tokens should come first when low_first=False."""
        result = tokens_length(texts, low_first=False)
        indices = [idx for idx, _ in result]
        assert indices == [2, 1, 0]

    def test_returns_correct_counts(self, texts):
        """Should return correct token counts."""
        result = tokens_length(texts, low_first=True)
        scores = {idx: count for idx, count in result}
        assert scores[0] == 1
        assert scores[1] == 2
        assert scores[2] == 4

    def test_custom_tokenizer(self):
        """Should use custom tokenizer when provided."""
        texts = ["a,b,c", "a,b"]
        result = tokens_length(texts, tokenizer=lambda s: s.split(","))
        indices = [idx for idx, _ in result]
        assert indices == [1, 0]  # 2 tokens, 3 tokens


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

    def test_invalid_scoring_raises(self, texts):
        """An unknown scoring mode should raise ValueError (no model load)."""
        with pytest.raises(ValueError, match="scoring must be"):
            perplexity_score(texts, scoring="bogus")

    def test_requires_model_and_tokenizer_together(self, texts):
        """Passing only one of model/tokenizer must raise, not silently pair a
        user object with a mismatched GPT-2 default. The check runs after the
        torch/transformers import, so skip if that stack isn't importable."""
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        try:
            from transformers import GPT2LMHeadModel  # noqa: F401
        except Exception:
            pytest.skip("transformers/torch too old to import GPT-2 classes")
        with pytest.raises(ValueError, match="both `model` and `tokenizer`"):
            perplexity_score(texts, model=object())
        with pytest.raises(ValueError, match="both `model` and `tokenizer`"):
            perplexity_score(texts, tokenizer=object())

    @pytest.mark.slow
    def test_log_likelihood_orders_typical_first(self, texts):
        """log_likelihood mode ranks by typicality too: normal text first."""
        result = perplexity_score(texts, low_first=True, scoring="log_likelihood")
        indices = [idx for idx, _ in result]
        assert indices[0] == 0

    @pytest.mark.slow
    def test_log_likelihood_normal_text_more_likely(self, texts):
        """Normal text should have higher total log-likelihood than nonsense."""
        result = perplexity_score(texts, scoring="log_likelihood")
        scores = {idx: ll for idx, ll in result}
        assert scores[0] > scores[2]

    @pytest.mark.slow
    def test_log_likelihood_short_text_gets_neg_infinity(self):
        """Unscorable text gets -inf (tail) under log_likelihood scoring."""
        result = perplexity_score(
            ["a", "The cat sat on the mat."], scoring="log_likelihood"
        )
        scores = {idx: ll for idx, ll in result}
        assert scores[0] == float("-inf")
        # ...and -inf sorts to the difficult tail, not the typical head.
        assert result[-1][0] == 0


class TestReadabilityScore:
    """Tests for readability_score function."""

    @pytest.fixture(autouse=True)
    def _ensure_nltk(self):
        """py-readability-metrics tokenizes via NLTK; ensure its data is present.

        NLTK 3.9 renamed ``punkt`` -> ``punkt_tab``. Download quietly if missing;
        skip these tests if the data can't be obtained (e.g. offline).
        """
        try:
            import nltk

            for res in ("punkt_tab", "punkt"):
                try:
                    nltk.data.find(f"tokenizers/{res}")
                except LookupError:
                    nltk.download(res, quiet=True)
        except Exception:
            pytest.skip("NLTK tokenizer data unavailable")

    @pytest.fixture
    def texts(self):
        # py-readability-metrics needs >= 100 words, so both passages are long.
        return [
            # Simple text (low grade level): short words, short sentences.
            "The cat sat on the mat. The dog ran in the yard. The sun was warm "
            "and bright. A boy and a girl went out to play. They ran and ran. "
            "Then they sat down to rest. A bird sang a song in the tree. The cat "
            "looked up at it. The dog wagged its tail. We had fun in the park all "
            "day long. Mom made us a good lunch. We ate bread and jam and eggs. "
            "The day was warm and nice. We went home when the sky grew dark. We "
            "were tired but glad. It was a fun day for all of us. We slept well "
            "that night and did not wake up.",
            # Complex text (high grade level): long words, long sentences.
            "The implementation of sophisticated algorithmic procedures "
            "necessitates a comprehensive understanding of computational "
            "complexity theory, abstract mathematical formalisms, and the "
            "intricate interdependencies among heterogeneous subsystems. "
            "Consequently, practitioners must assiduously evaluate the asymptotic "
            "ramifications of their architectural decisions, particularly when "
            "confronting the inherent tensions between theoretical optimality and "
            "pragmatic implementability. Furthermore, the proliferation of "
            "distributed paradigms introduces additional considerations regarding "
            "consistency, fault tolerance, and the probabilistic guarantees "
            "afforded by contemporary consensus mechanisms. Such considerations, "
            "when inadequately addressed, precipitate cascading failures whose "
            "etiology proves exceedingly difficult to diagnose, thereby "
            "undermining the reliability and maintainability of these "
            "increasingly elaborate computational artifacts. Moreover, the "
            "systematic verification of such guarantees demands formal "
            "methodologies whose computational expense frequently exceeds that "
            "of the very systems they purport to validate.",
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
        metrics = [
            "flesch_kincaid", "flesch", "gunning_fog", "coleman_liau", "ari",
            "dale_chall", "linsear_write",
        ]
        for metric in metrics:
            result = readability_score(texts, metric=metric)
            assert len(result) == 2
            # At least one passage is long enough to receive a real (finite)
            # score, so the metric's scoring branch actually runs.
            assert any(np.isfinite(score) for _, score in result)

    def test_smog_metric(self):
        """SMOG needs >= 30 sentences; give it a passage that long so its
        scoring branch runs instead of falling through to the inf handler."""
        long_text = " ".join(f"This is plain sentence number {i} here." for i in range(40))
        result = readability_score([long_text], metric="smog")
        assert len(result) == 1
        assert np.isfinite(result[0][1])

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


class TestEdgeCases:
    """Test edge cases for all text sorters."""

    def test_character_length_empty_list(self):
        """character_length should handle empty list."""
        assert character_length([]) == []

    def test_character_length_single_item(self):
        """character_length should handle single item."""
        result = character_length(["hello"])
        assert len(result) == 1
        assert result[0] == (0, 5)

    def test_tokens_length_empty_list(self):
        """tokens_length should handle empty list."""
        assert tokens_length([]) == []

    def test_tokens_length_single_item(self):
        """tokens_length should handle single item."""
        result = tokens_length(["hello world"])
        assert len(result) == 1
        assert result[0] == (0, 2)

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
