from __future__ import annotations

from collections.abc import Callable, Iterator
from difflib import SequenceMatcher


def simple_tokenizer(s: str) -> list[str]:
    return s.split()

def difflib_character_similarity(s1: str, s2: str) -> float:
    return SequenceMatcher(a=s1, b=s2).ratio()

def difflib_token_similarity(
    s1: str, s2: str, tokenizer: Callable[[str], list[str]] = simple_tokenizer
) -> float:
    seq1 = tokenizer(s1)
    seq2 = tokenizer(s2)
    return SequenceMatcher(a=seq1, b=seq2).ratio()

def _ngrams(tokens: list[str], n: int) -> Iterator[tuple[str, ...]]:
    """
    compute ngrams from list of tokens

    from:
    https://albertauyeung.github.io/2018/06/03/generating-ngrams.html/
    """
    assert n > 0
    ngrams = zip(*[tokens[i:] for i in range(n)], strict=False)
    return ngrams

def ngram_jaccard_similarity(
    text1: str,
    text2: str,
    n: int = 3,
    tokenizer: Callable[[str], list[str]] = simple_tokenizer,
) -> float:
    """
    Compute ngram (w/ jaccard) similarity between two lists of tokens.

    From Figure 5 (top) of:
    https://aclanthology.org/N19-1051.pdf
    """
    t1 = simple_tokenizer(text1)
    t2 = simple_tokenizer(text2)
    tally = 0
    for i in range(n):
        _n = i+1
        ngrams1 = set(_ngrams(t1, _n))
        ngrams2 = set(_ngrams(t2, _n))
        intersection = ngrams1.intersection(ngrams2)
        union = ngrams1.union(ngrams2)
        ratio = len(intersection) / float(len(union))
        tally += ratio
    score = tally / n
    return score

def ngram_jaccard_distance(t1: str, t2: str, n: int = 3) -> float:
    """
    Compute ngram (w/ jaccard) distance between two lists of tokens.

    From Figure 5 (top) of:
    https://aclanthology.org/N19-1051.pdf
    """
    sim = ngram_jaccard_similarity(t1, t2, n)
    dist = 1 - sim
    return dist
