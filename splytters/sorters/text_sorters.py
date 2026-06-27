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

    References:
        Splitting by length to test generalization (put long texts in test):
          - Søgaard, Ebert, Bastings & Filippova (2021), "We Need to Talk
            About Random Splits," EACL. Heuristic splits via a sentence-length
            threshold. https://aclanthology.org/2021.eacl-main.156
          - Varis & Bojar (2021), "Sequence Length is a Domain: Length-based
            Overfitting in Transformer Models," EMNLP.
            https://aclanthology.org/2021.emnlp-main.650
          - Lake & Baroni (2018), "Generalization without Systematicity," ICML,
            introduced the SCAN length split (train short, test long) -- a
            canonical length-generalization test (on output sequence length,
            synthetic data). https://arxiv.org/abs/1711.00350
          - Søgaard et al. trace the idea of testing generalization to longer
            sequences back to Siegelmann & Sontag (1992), "On the Computational
            Power of Neural Nets," COLT, pp. 440-449.
            https://doi.org/10.1145/130385.130432
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

    References:
        A token-count variant of length-based splitting; see ``character_length``
        for the motivation and references (Søgaard et al. 2021, EACL; Varis &
        Bojar 2021, EMNLP; Lake & Baroni 2018, ICML; Siegelmann & Sontag 1992,
        COLT).
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
    scoring: str = "perplexity",
) -> list[tuple[int, float]]:
    """
    Sort texts by how typical a language model finds them.

    A causal LM is run over each text to measure how "surprised" it is. Two
    scoring modes are offered (see ``scoring``); both rank the same way — typical
    text is ordered before unusual text — but they put different things in the
    difficult tail, so the returned score values differ.

    Useful for adversarial / curriculum splits: train on typical text, test on
    unusual text. Pair with :func:`splytters.sorted_stratified_split` to build
    "Likelihood Splits" (train on the high-likelihood head, test on the tail).

    Args:
        texts: list of strings to score
        model: HuggingFace causal LM (defaults to GPT-2 if None)
        tokenizer: HuggingFace tokenizer (defaults to GPT-2 if None)
        low_first: if True, typical/predictable texts first; if False,
            unusual/surprising texts first. The ordering is by *typicality*, so
            this holds for both scoring modes despite their opposite raw-value
            directions.
        scoring: how to score each text.
            - ``"perplexity"`` (default): ``exp`` of the mean per-token negative
              log-likelihood. Length-normalized; *lower* is more typical.
            - ``"log_likelihood"``: total (un-normalized) log-likelihood summed
              over the predicted tokens; *higher* is more typical. This is the
              score used by Likelihood Splits (Godbole & Jia, 2023).

    Returns:
        List of ``(index, score)`` tuples in the chosen metric's natural units,
        ordered by typicality per ``low_first``. Texts too short to score receive
        the most-difficult sentinel (``+inf`` perplexity / ``-inf``
        log-likelihood), so they always land in the tail.

    Raises:
        ValueError: if ``scoring`` is not ``"perplexity"`` or ``"log_likelihood"``.

    References:
        Godbole & Jia (2023), "Benchmarking Long-tail Generalization with
        Likelihood Splits," Findings of the ACL: EACL, pp. 963-983 —
        https://aclanthology.org/2023.findings-eacl.71 . They split datasets by
        LM likelihood (head -> train, tail -> test). Note they deliberately use
        *total log-likelihood* rather than perplexity: perplexity normalizes for
        length and over-corrects toward short examples in the tail (their fn. 3).
        Use ``scoring="log_likelihood"`` to reproduce their choice.
    """
    if scoring not in ("perplexity", "log_likelihood"):
        raise ValueError(
            f"scoring must be 'perplexity' or 'log_likelihood', got {scoring!r}"
        )

    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    if model is None or tokenizer is None:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.eval()

    device = next(model.parameters()).device

    # Texts too short to score sort into the difficult tail regardless of mode:
    # maximal perplexity (+inf) or minimal log-likelihood (-inf).
    unscorable = float("inf") if scoring == "perplexity" else float("-inf")

    scores = []
    with torch.no_grad():
        for i, text in enumerate(texts):
            encodings = tokenizer(text, return_tensors="pt").to(device)
            input_ids = encodings.input_ids

            if input_ids.size(1) < 2:
                # Too short to compute a per-token loss
                scores.append((i, unscorable))
                continue

            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()  # mean per-token negative log-likelihood
            if scoring == "perplexity":
                score = torch.exp(torch.tensor(loss)).item()
            else:
                n_tokens = input_ids.size(1) - 1  # number of predicted tokens
                score = -loss * n_tokens  # total log-likelihood
            scores.append((i, score))

    # Sort by difficulty (higher == more atypical/tail) so that low_first means
    # typical-first in both modes: high perplexity == low log-likelihood == hard.
    def difficulty(pair: tuple[int, float]) -> float:
        score = pair[1]
        return score if scoring == "perplexity" else -score

    scores.sort(key=difficulty, reverse=not low_first)
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
            else:  # pragma: no cover - unreachable; metric validated above
                raise ValueError(f"Unknown readability metric: {metric}")
            scores.append((i, score))
        except Exception:
            # Readability requires minimum text length
            scores.append((i, float("inf")))

    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


def gzip_complexity(
    texts: list[str], low_first: bool = True
) -> list[tuple[int, float]]:
    """
    Sort texts by gzip compression ratio — a model-free complexity proxy.

    Redundant / repetitive / simple text compresses well (low ratio); varied,
    information-dense text compresses poorly (high ratio). Needs no model, so
    unlike :func:`perplexity_score` it runs without any heavy dependency.

    Args:
        texts: list of strings.
        low_first: if True, the simplest (most compressible) texts first.

    Returns:
        List of (index, compression_ratio) tuples (compressed / raw bytes).

    Note:
        gzip carries ~18 bytes of header overhead, so the ratio is noisy for
        very short strings; it is most meaningful across texts of comparable
        length.
    """
    import gzip

    scores = []
    for i, text in enumerate(texts):
        raw = text.encode("utf-8")
        ratio = len(gzip.compress(raw)) / len(raw) if raw else 0.0
        scores.append((i, float(ratio)))
    scores.sort(key=lambda p: p[1], reverse=not low_first)
    return scores


