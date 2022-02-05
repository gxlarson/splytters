import copy
from pprint import pprint
from scipy.spatial.distance import euclidean as _dist_euclidean

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
    _texts = copy.deepcopy(texts)
    _texts.sort(key=lambda s: len(s), reverse=not short_first)
    return _texts


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
