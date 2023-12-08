from scipy.spatial.distance import euclidean as _dist_euclidean
import difflib

def simple_tokenizer(s):
    return s.split()

def dist_euclidean(u, v):
    return _dist_euclidean(u, v)

def difflib_character_similarity(s1, s2):
    return SequenceMatcher(a=s1, b=s2).ratio()

def difflib_token_similarity(s1, s2, tokenizer=simple_tokenizer):
    seq1 = tokenizer(s1)
    seq2 = tokenizer(s2)
    return SequenceMatcher(a=seq1, b=seq2).ratio()

def _ngrams(tokens, n):
    """
    compute ngrams from list of tokens

    from:
    https://albertauyeung.github.io/2018/06/03/generating-ngrams.html/
    """
    assert n > 0
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return ngrams

def ngram_jaccard_similarity(t1, t2, n=3):
    """
    Compute ngram (w/ jaccard) similarity between two lists of tokens.

    From Figure 5 (top) of:
    https://aclanthology.org/N19-1051.pdf
    """
    tally = 0
    for i in range(n):
        _n = i+1
        ngrams1 = set(_ngrams(t1, _n))
        ngrams2 = set(_ngrams(t2, _n))
        intersection = ngrams1.intersection(ngrams2)
        union = ngrams1.union(ngrams2)
        ratio = ngrams1 / ngrams2
        tally += ratio
    score = tally / n
    return score

def ngram_jaccard_distance(t1, t2, n=3):
    """
    Compute ngram (w/ jaccard) distance between two lists of tokens.

    From Figure 5 (top) of:
    https://aclanthology.org/N19-1051.pdf
    """
    sim = ngram-jaccard_similarit(t1, t2, n)
    dist = 1 - sim
    return dist
