"""
Adversarial splitting algorithms that minimize train-test similarity.

These methods create "hard" evaluation sets where test samples are
dissimilar from training samples, testing model generalization.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN, KMeans
from sklearn.utils import check_random_state

from splytters.utils import (
    as_index_array,
    cluster_embeddings,
    compute_centroid,
    resolve_n_train,
    validate_split_inputs,
)


def cluster_split(
    embeddings: ArrayLike,
    train_size: float | int = 0.7,
    method: str = "kmeans",
    n_clusters: int = 10,
    random_state: int = 42,
    **cluster_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split dataset by assigning entire clusters to train or test.

    Prevents 'cluster leakage' where similar samples end up on both sides.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        method: 'kmeans' or 'dbscan'
        n_clusters: number of clusters (kmeans only)
        random_state: for reproducibility
        **cluster_kwargs: passed to clustering algorithm

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set
    """
    embeddings = validate_split_inputs(embeddings, train_size)

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
    target_train = resolve_n_train(n_samples, train_size)

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

    return as_index_array(train_indices), as_index_array(test_indices)


def centroid_adversarial_split(
    embeddings: ArrayLike,
    train_size: float | int = 0.7,
    n_clusters: int = 10,
    random_state: int = 42,
    **cluster_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adversarial cluster split: assign clusters nearest to global centroid
    to train, furthest clusters to test.

    Combines centroid-distance ranking with cluster-based splitting to
    maximize train-test dissimilarity while keeping similar samples together.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        n_clusters: number of clusters
        random_state: for reproducibility
        **cluster_kwargs: passed to KMeans

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set
    """
    embeddings = validate_split_inputs(embeddings, train_size)

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
    target_train = resolve_n_train(n_samples, train_size)

    train_indices = []
    test_indices = []

    for cluster_id, _ in cluster_distances:
        indices = cluster_to_indices[cluster_id]
        if len(train_indices) + len(indices) <= target_train:
            train_indices.extend(indices)
        else:
            test_indices.extend(indices)

    return as_index_array(train_indices), as_index_array(test_indices)


def distance_adversarial_split(
    embeddings: ArrayLike,
    train_size: float | int = 0.7,
    metric: str = "euclidean",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adversarial split based on individual sample distance from centroid.

    Samples closest to centroid go to train, furthest go to test.
    Unlike cluster-based methods, this operates on individual samples.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        metric: distance metric
        random_state: accepted for API consistency; this split is deterministic

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set
    """
    embeddings = validate_split_inputs(embeddings, train_size)
    centroid = compute_centroid(embeddings)

    # Compute distance from centroid for each sample
    distances = np.linalg.norm(embeddings - centroid, axis=1)

    # Sort by distance (closest first)
    sorted_indices = np.argsort(distances)

    # Split
    n_train = resolve_n_train(len(embeddings), train_size)
    return as_index_array(sorted_indices[:n_train]), as_index_array(
        sorted_indices[n_train:]
    )


def density_adversarial_split(
    embeddings: ArrayLike,
    train_size: float | int = 0.7,
    metric: str = "euclidean",
    k: int = 10,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adversarial split based on local density.

    Samples in dense regions go to train, isolated samples go to test.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        metric: distance metric
        k: number of neighbors for density estimation
        random_state: accepted for API consistency; this split is deterministic

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set
    """
    embeddings = validate_split_inputs(embeddings, train_size)

    # TODO: Replace full pairwise matrix with NearestNeighbors(k) to reduce
    # memory from O(n²) to O(nk). Only k distances per point are needed here.
    distances = cdist(embeddings, embeddings, metric=metric)
    np.fill_diagonal(distances, np.inf)

    # Compute density as inverse of mean distance to k nearest neighbors
    k = min(k, len(embeddings) - 1)
    knn_distances = np.sort(distances, axis=1)[:, :k]
    densities = 1.0 / (knn_distances.mean(axis=1) + 1e-10)

    # Sort by density (highest density first -> train)
    sorted_indices = np.argsort(-densities)

    # Split
    n_train = resolve_n_train(len(embeddings), train_size)
    return as_index_array(sorted_indices[:n_train]), as_index_array(
        sorted_indices[n_train:]
    )


def outlier_adversarial_split(
    embeddings: ArrayLike,
    train_size: float | int = 0.7,
    contamination: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adversarial split using outlier detection.

    Normal samples go to train, outliers go to test.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        contamination: expected proportion of outliers
        random_state: for reproducibility

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set
    """
    from sklearn.ensemble import IsolationForest

    embeddings = validate_split_inputs(embeddings, train_size)

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
    n_train = resolve_n_train(len(embeddings), train_size)
    return as_index_array(sorted_indices[:n_train]), as_index_array(
        sorted_indices[n_train:]
    )


def min_cut_split(
    embeddings: ArrayLike,
    train_size: float | int = 0.7,
    similarity_threshold: float | None = None,
    metric: str = "euclidean",
    method: str = "spectral",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adversarial split using graph min-cut.

    Builds a similarity graph and finds a partition that minimizes
    edges (similarity) crossing the train/test boundary.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        similarity_threshold: only connect samples above this kernel similarity
                              (default: median similarity)
        metric: distance metric
        method: 'spectral' (fast, approximate) or 'stoer_wagner' (exact, slow)
        random_state: for reproducibility

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set
    """
    embeddings = validate_split_inputs(embeddings, train_size)
    n_samples = len(embeddings)

    if n_samples < 3:
        # Too few samples for meaningful split
        n_train = resolve_n_train(n_samples, train_size)
        return as_index_array(range(n_train)), as_index_array(
            range(n_train, n_samples)
        )

    # TODO: Build sparse similarity graph directly via kneighbors_graph instead
    # of materializing the full O(n²) distance matrix.
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
            # Sort and split to achieve desired train_size
            sorted_indices = np.argsort(fiedler)

        except Exception:
            # Fallback to random if eigendecomposition fails
            rng = check_random_state(random_state)
            sorted_indices = np.arange(n_samples)
            rng.shuffle(sorted_indices)

        n_train = resolve_n_train(n_samples, train_size)
        train_indices = sorted_indices[:n_train]
        test_indices = sorted_indices[n_train:]

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
                rng = check_random_state(random_state)
                rng.shuffle(main_component)

                n_train_needed = resolve_n_train(n_samples, train_size) - len(
                    train_indices
                )
                n_train_needed = max(0, min(n_train_needed, len(main_component)))

                train_indices.extend(main_component[:n_train_needed])
                test_indices.extend(main_component[n_train_needed:])

            else:
                # Run Stoer-Wagner
                cut_value, partition = nx.stoer_wagner(G)
                set1, set2 = list(partition[0]), list(partition[1])

                # Adjust to match train_size
                n_train = resolve_n_train(n_samples, train_size)

                if len(set1) >= n_train:
                    train_indices = set1[:n_train]
                    test_indices = set1[n_train:] + set2
                else:
                    train_indices = set1 + set2[:n_train - len(set1)]
                    test_indices = set2[n_train - len(set1):]

        except ImportError as err:
            raise ImportError(
                "networkx is required for method='stoer_wagner'"
            ) from err

    else:
        raise ValueError(f"Unknown method: {method}. Use 'spectral' or 'stoer_wagner'.")

    return as_index_array(train_indices), as_index_array(test_indices)


def normalized_cut_split(
    embeddings: ArrayLike,
    train_size: float | int = 0.7,
    metric: str = "euclidean",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adversarial split using normalized graph cut.

    Normalized cut balances the cut value with partition sizes,
    avoiding trivially small partitions.

    NCut(A,B) = cut(A,B)/vol(A) + cut(A,B)/vol(B)

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        metric: distance metric
        random_state: accepted for API consistency; this split is deterministic

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set
    """
    embeddings = validate_split_inputs(embeddings, train_size)
    n_samples = len(embeddings)

    if n_samples < 3:
        n_train = resolve_n_train(n_samples, train_size)
        return as_index_array(range(n_train)), as_index_array(
            range(n_train, n_samples)
        )

    # TODO: Build sparse similarity graph directly via kneighbors_graph instead
    # of materializing the full O(n²) distance matrix.
    distances = cdist(embeddings, embeddings, metric=metric)
    sigma = np.median(distances[distances > 0])
    if sigma == 0:
        sigma = 1.0
    W = np.exp(-distances**2 / (2 * sigma**2))
    np.fill_diagonal(W, 0)

    # Compute symmetric normalized Laplacian (D^{-1/2} W D^{-1/2}).
    D_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1) + 1e-10))

    L_norm = np.eye(n_samples) - D_inv_sqrt @ W @ D_inv_sqrt

    # Compute eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L_norm)

    # Use second eigenvector (Fiedler vector)
    fiedler = eigenvectors[:, 1]

    # Sort by Fiedler vector to get partition
    sorted_indices = np.argsort(fiedler)

    n_train = resolve_n_train(n_samples, train_size)
    return as_index_array(sorted_indices[:n_train]), as_index_array(
        sorted_indices[n_train:]
    )


def wasserstein_adversarial_split(
    embeddings: ArrayLike,
    train_size: float | int = 0.7,
    leaf_size: int = 40,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adversarial split via Wasserstein nearest-neighbors.

    Treats each embedding row as a 1-D distribution and selects, as the test
    set, the ``n_test`` points nearest (in 1-D Wasserstein / earth-mover
    distance) to a random anchor — yielding a tight, hard-to-generalize-to test
    neighborhood. Adapted from Søgaard et al., "We Need to Talk About Random
    Splits" (EACL 2021), https://aclanthology.org/2021.eacl-main.156 .

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        leaf_size: BallTree leaf size (higher = slower but less memory)
        random_state: for reproducibility (anchor sampling)

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set
    """
    from scipy.stats import wasserstein_distance
    from sklearn.neighbors import NearestNeighbors

    embeddings = validate_split_inputs(embeddings, train_size)
    n_samples = len(embeddings)
    n_test = n_samples - resolve_n_train(n_samples, train_size)
    n_test = max(1, min(n_test, n_samples - 1))
    rng = check_random_state(random_state)

    tree = NearestNeighbors(
        n_neighbors=n_test,
        algorithm="ball_tree",
        leaf_size=leaf_size,
        metric=wasserstein_distance,
    )
    tree.fit(embeddings)

    # Sample a random anchor in the per-dimension bounding box (works for any
    # real-valued embeddings, unlike the original integer-only formulation).
    anchor = rng.uniform(
        low=embeddings.min(axis=0), high=embeddings.max(axis=0)
    ).reshape(1, -1)

    test_indices = tree.kneighbors(anchor, return_distance=False)[0]
    test_set = {int(i) for i in test_indices}
    train_indices = [i for i in range(n_samples) if i not in test_set]
    return as_index_array(train_indices), as_index_array(test_indices)


def get_cluster_info(
    embeddings: ArrayLike,
    train_indices: ArrayLike,
    test_indices: ArrayLike,
    n_clusters: int = 10,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Utility to analyze cluster distribution across train/test split.

    Returns:
        dict with cluster statistics including leakage info
    """
    embeddings = np.asarray(embeddings)

    clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = clusterer.fit_predict(embeddings)

    train_set = {int(i) for i in train_indices}
    test_set = {int(i) for i in test_indices}

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
