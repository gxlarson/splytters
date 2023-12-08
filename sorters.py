import copy
from pprint import pprint
from readability import Readability
from scipy.spatial.distance import euclidean as _dist_euclidean

def simple_tokenizer(s):
    return s.split()

def dist_euclidean(u, v):
    return _dist_euclidean(u, v)

def distance_to_mean(embeddings, distance=dist_euclidean):
    """
    embeddings is of shape (n, d) where n is number of samples and d is dimensionality
    """
    (n, d) = embeddings.shape
    centroid = embeddings.mean(0)
    distances = []
    for i in range(n):
        dist = distance(embeddings[i], centroid)
        distances.append((i, dist))
    distances.sort(key=lambda p: p[1])
    return distances

def character_length(texts, short_first=True):
    """
    sorts texts based on character count
    """
    return _score_sorter(texts, len, short_first)

def tokens_length(texts, short_first=True, tokenizer=simple_tokenizer):
    """
    sorts texts based on number of tokens
    """
    data = [tokenizer(text) for text in texts]
    return _score_sorter(data, len, short_first)

# TODO
# def perplexity_score

# TODO
# def readability_score

def _score_sorter(data, score_fn, low_first=True):
    """
    scores data based on score_fn score
    """
    _data = copy.deepcopy(data)
    _data.sort(key=lambda d: score_fn(d), reverse=not(short_first))
    return _data

if __name__ == "__main__":
    """
    the code below sorts the texts based on distance to mean embedding

    a text that is farther away from the mean will appear later in the list
    """
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    texts = [
        "what is my balance",
        "my balance is what",
        "how much do I owe",
        "what's my balance"
    ]
    embeddings = embedder.encode(texts)
    distances = distance_to_mean(embeddings)
    distances = [(texts[i], d) for (i, d) in distances]
    pprint(distances)
