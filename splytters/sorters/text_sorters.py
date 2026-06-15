"""
Sorting algorithms for adversarial text dataset partitioning.

These functions rank text samples by various criteria (length, complexity,
perplexity, readability, vocabulary) to enable train-test splits that
maximize dissimilarity.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # only for annotations (PEP 563 keeps them un-evaluated)
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Heavy / optional dependencies (pysbd, torch, transformers, readability,
# wordfreq) are imported lazily inside the functions that use them, so the
# lightweight sorters (length, tokens, lexical diversity) work with no extras
# and a single broken/missing heavy dep can't block the whole module.


def simple_tokenizer(s: str) -> list[str]:
    """Split text on whitespace into a list of tokens."""
    return s.split()


def character_length(texts: list[str], low_first: bool = True) -> list[tuple[int, int]]:
    """
    Sort texts by character count.

    Args:
        texts: list of strings
        low_first: if True, shortest texts first; if False, longest first

    Returns:
        List of (index, character_count) tuples sorted by length.
    """
    scores = [(i, len(text)) for i, text in enumerate(texts)]
    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


def tokens_length(
    texts: list[str],
    low_first: bool = True,
    tokenizer: Callable[[str], list[str]] = simple_tokenizer,
) -> list[tuple[int, int]]:
    """
    Sort texts by token count.

    Args:
        texts: list of strings
        low_first: if True, fewest tokens first; if False, most tokens first
        tokenizer: function that splits text into tokens, default whitespace split

    Returns:
        List of (index, token_count) tuples sorted by token count.
    """
    scores = [(i, len(tokenizer(text))) for i, text in enumerate(texts)]
    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


def sentence_count(
    texts: list[str], language: str = "en", low_first: bool = True
) -> list[tuple[int, int]]:
    """
    Sort texts by number of sentences.

    Uses pysbd for robust sentence boundary detection across languages.

    Useful for adversarial splits: train on single-sentence texts,
    test on multi-sentence/complex texts.

    Args:
        texts: list of strings
        language: language code for sentence segmentation (default 'en')
        low_first: if True, fewer sentences first; if False, more sentences first

    Returns:
        List of (index, sentence_count) tuples sorted by sentence count.
    """
    import pysbd

    segmenter = pysbd.Segmenter(language=language, clean=False)

    scores = []
    for i, text in enumerate(texts):
        sentences = segmenter.segment(text)
        scores.append((i, len(sentences)))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


def lexical_diversity(
    texts: list[str],
    tokenizer: Callable[[str], list[str]] = simple_tokenizer,
    low_first: bool = True,
) -> list[tuple[int, float]]:
    """
    Sort texts by lexical diversity (type-token ratio).

    Type-token ratio = unique tokens / total tokens.
    Higher values indicate more diverse vocabulary; lower values indicate
    more repetitive text.

    Useful for adversarial splits: train on repetitive/simple vocabulary,
    test on diverse/rich vocabulary.

    Args:
        texts: list of strings
        tokenizer: function that splits text into tokens
        low_first: if True, repetitive texts first; if False, diverse texts first

    Returns:
        List of (index, ttr) tuples sorted by type-token ratio.
        Texts with no tokens receive a score of 0.
    """
    scores = []
    for i, text in enumerate(texts):
        tokens = tokenizer(text)
        if len(tokens) == 0:
            scores.append((i, 0.0))
        else:
            ttr = len(set(tokens)) / len(tokens)
            scores.append((i, ttr))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


def vocabulary_rarity(
    texts: list[str],
    language: str = "en",
    tokenizer: Callable[[str], list[str]] = simple_tokenizer,
    low_first: bool = True,
) -> list[tuple[int, float]]:
    """
    Sort texts by average word rarity.

    Uses word frequency data to score each word's rarity. Rarer words have
    lower frequency, so we use (1 - frequency) as the rarity score.

    Useful for adversarial splits: train on common vocabulary,
    test on rare/specialized vocabulary.

    Args:
        texts: list of strings
        language: language code for frequency lookup (default 'en')
        tokenizer: function that splits text into tokens
        low_first: if True, common vocabulary first; if False, rare vocabulary first

    Returns:
        List of (index, avg_rarity) tuples sorted by average word rarity.
        Texts with no tokens receive a score of 0.
    """
    from wordfreq import word_frequency

    scores = []
    for i, text in enumerate(texts):
        tokens = tokenizer(text.lower())
        if len(tokens) == 0:
            scores.append((i, 0.0))
            continue

        # Calculate average rarity (1 - frequency) for all tokens
        # word_frequency returns 0 for unknown words, so rarity = 1 for unknown
        rarities = [1.0 - word_frequency(token, language) for token in tokens]
        avg_rarity = sum(rarities) / len(rarities)
        scores.append((i, avg_rarity))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


def perplexity_score(
    texts: list[str],
    model: GPT2LMHeadModel | None = None,
    tokenizer: GPT2TokenizerFast | None = None,
    low_first: bool = True,
) -> list[tuple[int, float]]:
    """
    Sort texts by language model perplexity.

    Perplexity measures how "surprised" a language model is by the text.
    Lower perplexity indicates more typical/predictable text, higher
    perplexity indicates unusual or difficult text.

    Useful for adversarial splits: train on typical text, test on unusual text.

    Args:
        texts: list of strings to score
        model: HuggingFace causal LM (defaults to GPT-2 if None)
        tokenizer: HuggingFace tokenizer (defaults to GPT-2 if None)
        low_first: if True, typical/predictable texts first;
                   if False, unusual/surprising texts first

    Returns:
        List of (index, perplexity) tuples sorted by perplexity.
        Texts too short to score receive perplexity of infinity.
    """
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    if model is None or tokenizer is None:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()

    device = next(model.parameters()).device
    scores = []

    with torch.no_grad():
        for i, text in enumerate(texts):
            encodings = tokenizer(text, return_tensors="pt").to(device)
            input_ids = encodings.input_ids

            if input_ids.size(1) < 2:
                # Too short to compute perplexity
                scores.append((i, float("inf")))
                continue

            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
            perplexity = torch.exp(torch.tensor(loss)).item()
            scores.append((i, perplexity))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


def readability_score(
    texts: list[str], metric: str = "flesch_kincaid", low_first: bool = True
) -> list[tuple[int, float]]:
    """
    Sort texts by readability score.

    Readability scores estimate the education level required to understand text.
    Higher scores generally indicate more complex/difficult text.

    Useful for adversarial splits: train on simple text, test on complex text.

    Args:
        texts: list of strings to score
        metric: readability formula to use, one of:
            - 'flesch_kincaid': Flesch-Kincaid Grade Level
            - 'flesch': Flesch Reading Ease (higher = easier, inverted scale)
            - 'gunning_fog': Gunning Fog Index
            - 'coleman_liau': Coleman-Liau Index
            - 'dale_chall': Dale-Chall Readability Score
            - 'ari': Automated Readability Index
            - 'linsear_write': Linsear Write Formula
            - 'smog': SMOG Index
        low_first: if True, easier/simpler texts first;
                   if False, harder/complex texts first

    Returns:
        List of (index, score) tuples sorted by readability.
        Texts too short to score receive a score of infinity.
    """
    valid_metrics = {
        "flesch_kincaid", "flesch", "gunning_fog", "coleman_liau",
        "dale_chall", "ari", "linsear_write", "smog",
    }
    # Validate up front so an unknown metric raises instead of being swallowed
    # by the per-text "too short -> inf" exception handler below.
    if metric not in valid_metrics:
        raise ValueError(f"Unknown readability metric: {metric}")

    from readability import Readability

    scores = []

    for i, text in enumerate(texts):
        try:
            r = Readability(text)
            if metric == "flesch_kincaid":
                score = r.flesch_kincaid().score
            elif metric == "flesch":
                score = r.flesch().score
            elif metric == "gunning_fog":
                score = r.gunning_fog().score
            elif metric == "coleman_liau":
                score = r.coleman_liau().score
            elif metric == "dale_chall":
                score = r.dale_chall().score
            elif metric == "ari":
                score = r.ari().score
            elif metric == "linsear_write":
                score = r.linsear_write().score
            elif metric == "smog":
                score = r.smog().score
            else:
                raise ValueError(f"Unknown readability metric: {metric}")
            scores.append((i, score))
        except Exception:
            # Readability requires minimum text length
            scores.append((i, float("inf")))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


