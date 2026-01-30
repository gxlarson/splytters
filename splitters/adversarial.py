"""
Adversarial splitting algorithms that minimize train-test similarity.

These methods create "hard" evaluation sets where test samples are
dissimilar from training samples, testing model generalization.
"""

from collections import defaultdict

import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, DBSCAN

from splitters.utils import cluster_embeddings, compute_centroid


def cluster_split(
    embeddings,
    train_ratio=0.7,
    method="kmeans",
    n_clusters=10,
    random_state=42,
    **cluster_kwargs
):
    """
    Split dataset by assigning entire clusters to train or test.

    Prevents 'cluster leakage' where similar samples end up on both sides.

    Args:
        embeddings: np.array of shape (n_samples, embedding_dim)
        train_ratio: fraction of data for training
        method: 'kmeans' or 'dbscan'
        n_clusters: number of clusters (kmeans only)
        random_state: for reproducibility
        **cluster_kwargs: passed to clustering algorithm

    Returns:
        train_indices: list of indices for training set
        test_indices: list of indices for test set
    """
    embeddings = np.asarray(embeddings)

    if method == "kmeans":
        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto",
            **cluster_kwargs
        )
    elif method == "dbscan":
        clusterer = DBSCAN(**cluster_kwargs)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    labels = clusterer.fit_predict(embeddings)

    # Group indices by cluster
    cluster_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_to_indices[label].append(idx)

    # Sort clusters by size (larger clusters assigned first for better ratio)
    clusters_by_size = sorted(
        cluster_to_indices.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    # Greedily assign clusters to train until we hit target ratio
    n_samples = len(embeddings)
    target_train = int(n_samples * train_ratio)

    train_indices = []
    test_indices = []

    for cluster_id, indices in clusters_by_size:
        # DBSCAN noise points (label=-1) go to test set
        if cluster_id == -1:
            test_indices.extend(indices)
            continue

        if len(train_indices) + len(indices) <= target_train:
            train_indices.extend(indices)
        else:
            test_indices.extend(indices)

    return train_indices, test_indices


def centroid_adversarial_split(
    embeddings,
    train_ratio=0.7,
    n_clusters=10,
    random_state=42,
    **cluster_kwargs
):
    """
    Adversarial cluster split: assign clusters nearest to global centroid
    to train, furthest clusters to test.

    Combines centroid-distance ranking with cluster-based splitting to
    maximize train-test dissimilarity while keeping similar samples together.

    Args:
        embeddings: np.array of shape (n_samples, embedding_dim)
        train_ratio: fraction of data for training
        n_clusters: number of clusters
        random_state: for reproducibility
        **cluster_kwargs: passed to KMeans

    Returns:
        train_indices: list of indices for training set
        test_indices: list of indices for test set
    """
    embeddings = np.asarray(embeddings)

    labels, cluster_to_indices, cluster_centers = cluster_embeddings(
        embeddings, n_clusters, "kmeans", random_state, **cluster_kwargs
    )

    # Compute global centroid
    global_centroid = compute_centroid(embeddings)

    # Rank clusters by distance from global centroid
    cluster_distances = []
    for i, center in enumerate(cluster_centers):
        dist = np.linalg.norm(center - global_centroid)
        cluster_distances.append((i, dist))

    # Sort by distance (closest first -> train, furthest -> test)
    cluster_distances.sort(key=lambda x: x[1])

    # Assign clusters to train/test based on distance ranking
    n_samples = len(embeddings)
    target_train = int(n_samples * train_ratio)

    train_indices = []
    test_indices = []

    for cluster_id, _ in cluster_distances:
        indices = cluster_to_indices[cluster_id]
        if len(train_indices) + len(indices) <= target_train:
            train_indices.extend(indices)
        else:
            test_indices.extend(indices)

    return train_indices, test_indices


def distance_adversarial_split(embeddings, train_ratio=0.7, metric="euclidean"):
    """
    Adversarial split based on individual sample distance from centroid.

    Samples closest to centroid go to train, furthest go to test.
    Unlike cluster-based methods, this operates on individual samples.

    Args:
        embeddings: np.array of shape (n_samples, embedding_dim)
        train_ratio: fraction of data for training
        metric: distance metric

    Returns:
        train_indices: list of indices for training set
        test_indices: list of indices for test set
    """
    embeddings = np.asarray(embeddings)
    centroid = compute_centroid(embeddings)

    # Compute distance from centroid for each sample
    distances = np.linalg.norm(embeddings - centroid, axis=1)

    # Sort by distance (closest first)
    sorted_indices = np.argsort(distances)

    # Split
    n_train = int(len(embeddings) * train_ratio)
    train_indices = sorted_indices[:n_train].tolist()
    test_indices = sorted_indices[n_train:].tolist()

    return train_indices, test_indices


def density_adversarial_split(
    embeddings,
    train_ratio=0.7,
    metric="euclidean",
    k=10
):
    """
    Adversarial split based on local density.

    Samples in dense regions go to train, isolated samples go to test.

    Args:
        embeddings: np.array of shape (n_samples, embedding_dim)
        train_ratio: fraction of data for training
        metric: distance metric
        k: number of neighbors for density estimation

    Returns:
        train_indices: list of indices for training set
        test_indices: list of indices for test set
    """
    embeddings = np.asarray(embeddings)

    # Compute pairwise distances
    distances = cdist(embeddings, embeddings, metric=metric)
    np.fill_diagonal(distances, np.inf)

    # Compute density as inverse of mean distance to k nearest neighbors
    k = min(k, len(embeddings) - 1)
    knn_distances = np.sort(distances, axis=1)[:, :k]
    densities = 1.0 / (knn_distances.mean(axis=1) + 1e-10)

    # Sort by density (highest density first -> train)
    sorted_indices = np.argsort(-densities)

    # Split
    n_train = int(len(embeddings) * train_ratio)
    train_indices = sorted_indices[:n_train].tolist()
    test_indices = sorted_indices[n_train:].tolist()

    return train_indices, test_indices


def outlier_adversarial_split(
    embeddings,
    train_ratio=0.7,
    contamination=0.1,
    random_state=42
):
    """
    Adversarial split using outlier detection.

    Normal samples go to train, outliers go to test.

    Args:
        embeddings: np.array of shape (n_samples, embedding_dim)
        train_ratio: fraction of data for training
        contamination: expected proportion of outliers
        random_state: for reproducibility

    Returns:
        train_indices: list of indices for training set
        test_indices: list of indices for test set
    """
    from sklearn.ensemble import IsolationForest

    embeddings = np.asarray(embeddings)

    detector = IsolationForest(
        contamination=contamination,
        random_state=random_state
    )
    detector.fit(embeddings)

    # Get outlier scores (more negative = more normal)
    scores = detector.score_samples(embeddings)

    # Sort by score (most normal first -> train)
    sorted_indices = np.argsort(-scores)

    # Split
    n_train = int(len(embeddings) * train_ratio)
    train_indices = sorted_indices[:n_train].tolist()
    test_indices = sorted_indices[n_train:].tolist()

    return train_indices, test_indices


def min_cut_split(
    embeddings,
    train_ratio=0.7,
    similarity_threshold=None,
    metric="euclidean",
    method="spectral",
    random_state=42,
):
    """
    Adversarial split using graph min-cut.

    Builds a similarity graph and finds a partition that minimizes
    edges (similarity) crossing the train/test boundary.

    Args:
        embeddings: np.array of shape (n_samples, embedding_dim)
        train_ratio: fraction of data for training
        similarity_threshold: only connect samples above this similarity
                              (default: median similarity)
        metric: distance metric
        method: 'spectral' (fast, approximate) or 'stoer_wagner' (exact, slow)
        random_state: for reproducibility

    Returns:
        train_indices: list of indices for training set
        test_indices: list of indices for test set
    """
    embeddings = np.asarray(embeddings)
    n_samples = len(embeddings)

    if n_samples < 3:
        # Too few samples for meaningful split
        n_train = int(n_samples * train_ratio)
        return list(range(n_train)), list(range(n_train, n_samples))

    # Compute pairwise distances and convert to similarities
    distances = cdist(embeddings, embeddings, metric=metric)

    # Convert distance to similarity using Gaussian kernel
    sigma = np.median(distances[distances > 0])
    if sigma == 0:
        sigma = 1.0
    similarities = np.exp(-distances**2 / (2 * sigma**2))
    np.fill_diagonal(similarities, 0)  # No self-loops

    # Threshold to create sparse graph
    if similarity_threshold is None:
        nonzero_sims = similarities[similarities > 0]
        if len(nonzero_sims) > 0:
            similarity_threshold = np.median(nonzero_sims)
        else:
            similarity_threshold = 0.0

    similarities[similarities < similarity_threshold] = 0

    if method == "spectral":
        # Spectral partitioning using Fiedler vector
        # (eigenvector corresponding to 2nd smallest eigenvalue of Laplacian)

        L = laplacian(csr_matrix(similarities), normed=True)

        try:
            # Get 2 smallest eigenvalues/vectors
            eigenvalues, eigenvectors = eigsh(L, k=2, which='SM', tol=1e-6)

            # Fiedler vector (2nd eigenvector)
            fiedler = eigenvectors[:, 1]

            # Partition by Fiedler vector values
            # Sort and split to achieve desired train_ratio
            sorted_indices = np.argsort(fiedler)

        except Exception:
            # Fallback to random if eigendecomposition fails
            rng = np.random.RandomState(random_state)
            sorted_indices = np.arange(n_samples)
            rng.shuffle(sorted_indices)

        n_train = int(n_samples * train_ratio)
        train_indices = sorted_indices[:n_train].tolist()
        test_indices = sorted_indices[n_train:].tolist()

    elif method == "stoer_wagner":
        # Exact min-cut using Stoer-Wagner algorithm (slower)
        try:
            import networkx as nx

            # Build weighted graph
            G = nx.Graph()
            G.add_nodes_from(range(n_samples))

            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if similarities[i, j] > 0:
                        G.add_edge(i, j, weight=similarities[i, j])

            # Check if graph is connected
            if not nx.is_connected(G):
                # Find connected components and use largest
                components = list(nx.connected_components(G))
                components.sort(key=len, reverse=True)

                # Assign smaller components to test, sample from largest for train
                train_indices = []
                test_indices = []

                for comp in components[1:]:
                    test_indices.extend(comp)

                main_component = list(components[0])
                rng = np.random.RandomState(random_state)
                rng.shuffle(main_component)

                n_train_needed = int(n_samples * train_ratio) - len(train_indices)
                n_train_needed = max(0, min(n_train_needed, len(main_component)))

                train_indices.extend(main_component[:n_train_needed])
                test_indices.extend(main_component[n_train_needed:])

            else:
                # Run Stoer-Wagner
                cut_value, partition = nx.stoer_wagner(G)
                set1, set2 = list(partition[0]), list(partition[1])

                # Adjust to match train_ratio
                n_train = int(n_samples * train_ratio)

                if len(set1) >= n_train:
                    train_indices = set1[:n_train]
                    test_indices = set1[n_train:] + set2
                else:
                    train_indices = set1 + set2[:n_train - len(set1)]
                    test_indices = set2[n_train - len(set1):]

        except ImportError:
            raise ImportError("networkx is required for method='stoer_wagner'")

    else:
        raise ValueError(f"Unknown method: {method}. Use 'spectral' or 'stoer_wagner'.")

    return train_indices, test_indices


def normalized_cut_split(
    embeddings,
    train_ratio=0.7,
    metric="euclidean",
    random_state=42,
):
    """
    Adversarial split using normalized graph cut.

    Normalized cut balances the cut value with partition sizes,
    avoiding trivially small partitions.

    NCut(A,B) = cut(A,B)/vol(A) + cut(A,B)/vol(B)

    Args:
        embeddings: np.array of shape (n_samples, embedding_dim)
        train_ratio: fraction of data for training
        metric: distance metric
        random_state: for reproducibility

    Returns:
        train_indices: list of indices for training set
        test_indices: list of indices for test set
    """
    embeddings = np.asarray(embeddings)
    n_samples = len(embeddings)

    if n_samples < 3:
        n_train = int(n_samples * train_ratio)
        return list(range(n_train)), list(range(n_train, n_samples))

    # Compute similarity matrix
    distances = cdist(embeddings, embeddings, metric=metric)
    sigma = np.median(distances[distances > 0])
    if sigma == 0:
        sigma = 1.0
    W = np.exp(-distances**2 / (2 * sigma**2))
    np.fill_diagonal(W, 0)

    # Compute normalized Laplacian
    D = np.diag(W.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1) + 1e-10))

    L_norm = np.eye(n_samples) - D_inv_sqrt @ W @ D_inv_sqrt

    # Compute eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L_norm)

    # Use second eigenvector (Fiedler vector)
    fiedler = eigenvectors[:, 1]

    # Sort by Fiedler vector to get partition
    sorted_indices = np.argsort(fiedler)

    n_train = int(n_samples * train_ratio)
    train_indices = sorted_indices[:n_train].tolist()
    test_indices = sorted_indices[n_train:].tolist()

    return train_indices, test_indices


def get_cluster_info(embeddings, train_indices, test_indices, n_clusters=10, random_state=42):
    """
    Utility to analyze cluster distribution across train/test split.

    Returns:
        dict with cluster statistics including leakage info
    """
    embeddings = np.asarray(embeddings)

    clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = clusterer.fit_predict(embeddings)

    train_set = set(train_indices)
    test_set = set(test_indices)

    cluster_stats = {}
    for cluster_id in range(n_clusters):
        cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
        in_train = sum(1 for i in cluster_indices if i in train_set)
        in_test = sum(1 for i in cluster_indices if i in test_set)
        cluster_stats[cluster_id] = {
            "total": len(cluster_indices),
            "in_train": in_train,
            "in_test": in_test,
            "leakage": min(in_train, in_test) > 0
        }

    total_leaking = sum(1 for s in cluster_stats.values() if s["leakage"])

    return {
        "cluster_stats": cluster_stats,
        "clusters_with_leakage": total_leaking,
        "leakage_ratio": total_leaking / n_clusters
    }
