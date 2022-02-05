from statistics import mean
from distances import dist_euclidean

def mean_dist(embeddings, distance=dist_euclidean):
    (n, d) = embeddings.shape
    centroid = embeddings.mean(0)
    distances = []
    for i in range(n):
        distances.append(distance(centroid, embeddings[i]))
    return mean(distances)
