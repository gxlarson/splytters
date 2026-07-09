"""
Adversarial splitting algorithms that minimize train-test similarity.

These methods create "hard" evaluation sets where test samples are
dissimilar from training samples, testing model generalization.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigh as dense_eigh
from scipy.sparse.csgraph import laplacian
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.utils import check_random_state

from splytters.utils import (
    as_index_array,
    compute_centroid,
    optimized_split,
    resolve_n_train,
    validate_split_inputs,
)

# Below this test-set fraction, minority_split's result is effectively
# degenerate (clusters are nearly label-pure): warn rather than fail silently.
_MINORITY_DEGENERATE_FRACTION = 0.05


def _cluster_centroids(
    embeddings: np.ndarray, cluster_to_indices: dict[int, list[int]]
) -> dict[int, np.ndarray]:
    """Mean embedding (centroid) of each cluster's members."""
    return {cid: embeddings[idxs].mean(axis=0) for cid, idxs in cluster_to_indices.items()}


def _assign_by_size(
    cluster_to_indices: dict[int, list[int]], target_train: int
) -> tuple[list[int], list[int]]:
    """Greedily fill train with the largest clusters first (DBSCAN noise -> test)."""
    clusters_by_size = sorted(
        cluster_to_indices.items(), key=lambda kv: len(kv[1]), reverse=True
    )
    train: list[int] = []
    test: list[int] = []
    for cluster_id, indices in clusters_by_size:
        if cluster_id == -1:  # DBSCAN noise points -> test
            test.extend(indices)
        elif len(train) + len(indices) <= target_train:
            train.extend(indices)
        else:
            test.extend(indices)
    return train, test


def _assign_by_centroid(
    embeddings: np.ndarray, cluster_to_indices: dict[int, list[int]], target_train: int
) -> tuple[list[int], list[int]]:
    """Rank clusters by distance from the global centroid; near -> train, far -> test."""
    global_centroid = compute_centroid(embeddings)
    centroids = _cluster_centroids(embeddings, cluster_to_indices)
    ranked = sorted(
        cluster_to_indices, key=lambda cid: np.linalg.norm(centroids[cid] - global_centroid)
    )
    train: list[int] = []
    test: list[int] = []
    for cid in ranked:
        indices = cluster_to_indices[cid]
        if len(train) + len(indices) <= target_train:
            train.extend(indices)
        else:
            test.extend(indices)
    return train, test


def _assign_closest(
    embeddings: np.ndarray, cluster_to_indices: dict[int, list[int]], target_test: int
) -> tuple[list[int], list[int]]:
    """CLOSEST-SPLIT: build one coherent, isolated test pocket in latent space.

    Seed with the most isolated cluster (largest mean cosine distance to the
    other cluster centroids), then grow the test set by repeatedly adding the
    remaining cluster nearest (cosine) to the current test pocket, stopping
    before the target test size is exceeded.
    """
    cids = list(cluster_to_indices)
    centroids = _cluster_centroids(embeddings, cluster_to_indices)
    sizes = [len(cluster_to_indices[cid]) for cid in cids]

    # Pairwise cosine distances between cluster centroids (rows/cols aligned to cids).
    C = np.vstack([centroids[cid] for cid in cids])
    cos = cdist(C, C, metric="cosine")
    isolation = cos.mean(axis=1)  # mean distance from each cluster to all others

    # Seed: most isolated cluster that fits the target (else just the most isolated).
    order = sorted(range(len(cids)), key=lambda j: isolation[j], reverse=True)
    seed = next((j for j in order if sizes[j] <= target_test), order[0])

    test_pos = {seed}
    test_size = sizes[seed]
    while test_size < target_test:
        remaining = [j for j in range(len(cids)) if j not in test_pos]
        if not remaining:
            break
        members = list(test_pos)
        nearest = min(remaining, key=lambda j: cos[j, members].mean())
        if test_size + sizes[nearest] > target_test:
            break
        test_pos.add(nearest)
        test_size += sizes[nearest]

    test_cids = {cids[j] for j in test_pos}
    train: list[int] = []
    test: list[int] = []
    for cid, indices in cluster_to_indices.items():
        (test if cid in test_cids else train).extend(indices)
    return train, test


def _assign_subset_sum(
    cluster_to_indices: dict[int, list[int]],
    y: np.ndarray,
    target_test: int,
    n_samples: int,
) -> tuple[list[int], list[int]]:
    """SUBSET-SUM-SPLIT: pick clusters whose pooled per-class counts best match a
    class-balanced target test set.

    Each cluster contributes a per-class count vector; we greedily select the
    subset of clusters whose summed vector is closest (L1) to the target vector
    ``target_test * class_proportions`` -- an approximate multidimensional
    subset-sum keeping the test set's label distribution close to the whole.
    """
    classes = np.unique(y)
    class_idx = {c: k for k, c in enumerate(classes)}

    cids = list(cluster_to_indices)
    vecs: dict[int, np.ndarray] = {}
    for cid, idxs in cluster_to_indices.items():
        v = np.zeros(len(classes))
        labels_here, counts = np.unique(y[idxs], return_counts=True)
        for c, cnt in zip(labels_here, counts, strict=True):
            v[class_idx[c]] = cnt
        vecs[cid] = v

    global_counts = np.array([np.sum(y == c) for c in classes], dtype=float)
    target = target_test * global_counts / n_samples

    selected: list[int] = []
    current = np.zeros(len(classes))
    remaining = set(cids)
    best_dist = float(np.abs(current - target).sum())
    while remaining:
        cid = min(remaining, key=lambda c: float(np.abs(current + vecs[c] - target).sum()))
        new_dist = float(np.abs(current + vecs[cid] - target).sum())
        if new_dist >= best_dist:
            break
        selected.append(cid)
        current = current + vecs[cid]
        best_dist = new_dist
        remaining.discard(cid)

    if not selected:  # fall back to the single best-matching cluster
        selected = [min(cids, key=lambda c: float(np.abs(vecs[c] - target).sum()))]

    test_cids = set(selected)
    train: list[int] = []
    test: list[int] = []
    for cid, indices in cluster_to_indices.items():
        (test if cid in test_cids else train).extend(indices)
    return train, test


def cluster_split(
    embeddings: ArrayLike,
    train_size: float | int = 0.7,
    method: str = "kmeans",
    n_clusters: int = 10,
    random_state: int = 42,
    *,
    strategy: str = "size",
    y: ArrayLike | None = None,
    **cluster_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split dataset by assigning entire clusters to train or test.

    Prevents 'cluster leakage' where similar samples end up on both sides. The
    embeddings are clustered (``method``), then whole clusters are assigned to
    train/test according to ``strategy``.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        method: clustering algorithm, 'kmeans' or 'dbscan'
        n_clusters: number of clusters (kmeans only)
        random_state: for reproducibility
        strategy: cluster -> train/test assignment policy:
            - ``"size"`` (default): greedily fill train with the largest clusters
              until the target ratio is met. The original, target-ratio-driven
              behavior; DBSCAN noise points go to test.
            - ``"centroid"``: rank clusters by distance from the global centroid,
              nearest -> train, farthest -> test (adversarial). This is what
              :func:`centroid_adversarial_split` delegates to.
            - ``"subset_sum"``: select a subset of clusters whose pooled per-class
              counts best match a class-balanced target test set. Requires ``y``.
            - ``"closest"``: seed the most isolated cluster and grow it by
              nearest-neighbor into a single coherent test "pocket" (adversarial).
        y: class labels of shape (n_samples,). Required for ``strategy="subset_sum"``;
            ignored by the other strategies.
        **cluster_kwargs: passed to the clustering algorithm

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set

    Raises:
        ValueError: on an unknown ``method`` or ``strategy``, or if
            ``strategy="subset_sum"`` is used without a valid ``y``.

    References:
        The ``"subset_sum"`` and ``"closest"`` strategies implement SUBSET-SUM-SPLIT
        and CLOSEST-SPLIT from Züfle, Dankers & Titov (2023), "Latent Feature-based
        Data Splits to Improve Generalisation Evaluation," GenBench Workshop @ EMNLP,
        pp. 112-129. https://aclanthology.org/2023.genbench-1.9 . They cluster a
        model's hidden representations and assign whole clusters to test to expose
        latent-space "blind spots"; their analysis finds that the resulting
        difficulty does not correlate with surface-level properties (e.g. length),
        complementing the surface-based :mod:`splytters.sorters`.

        The label-balanced idea behind ``"subset_sum"`` traces to the earlier
        ClusterDataSplit of Wecker, Friedrich & Adel (2020), "ClusterDataSplit:
        Exploring Challenging Clustering-Based Data Splits for Model Performance
        Evaluation," Eval4NLP @ COLING, which clusters data into splits that
        differ lexically from train while keeping the label distribution fixed.
        https://aclanthology.org/2020.eval4nlp-1.15 . See :func:`cluster_kfold`
        for their cross-validation variant.

        Napoli & White (2025), "Clustering-Based Validation Splits for Model
        Selection under Domain Shift," TMLR, ground this family theoretically:
        maximizing the MMD between the two sets is equivalent to kernel k-means
        clustering (k=2), and they replace ClusterDataSplit's greedy
        class-balancing with a linear program (with convergence guarantees).
        https://openreview.net/forum?id=Q692C0WtiD . See
        :func:`mmd_maximized_split` for the MMD-maximizing objective.

    Seed stability: varies with the seed -- the KMeans clustering, and which
    whole clusters are held out, shift with the seed, so the test set can differ
    substantially between runs (from near-identical to nearly disjoint) even
    though the clusters themselves are similar.
    """
    embeddings = validate_split_inputs(embeddings, train_size)

    valid_strategies = {"size", "centroid", "subset_sum", "closest"}
    if strategy not in valid_strategies:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. Choose from {sorted(valid_strategies)}"
        )

    n_samples = len(embeddings)

    if strategy == "subset_sum":
        if y is None:
            raise ValueError("strategy='subset_sum' requires class labels `y`")
        y = np.asarray(y)
        if len(y) != n_samples:
            raise ValueError(
                f"y has length {len(y)} but embeddings has {n_samples} rows"
            )

    if method == "kmeans":
        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto",
            **cluster_kwargs,
        )
    elif method == "dbscan":
        clusterer = DBSCAN(**cluster_kwargs)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    labels = clusterer.fit_predict(embeddings)
    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_to_indices[int(label)].append(idx)

    target_train = resolve_n_train(n_samples, train_size)
    target_test = n_samples - target_train

    if strategy == "size":
        train, test = _assign_by_size(cluster_to_indices, target_train)
    elif strategy == "centroid":
        train, test = _assign_by_centroid(embeddings, cluster_to_indices, target_train)
    elif strategy == "closest":
        train, test = _assign_closest(embeddings, cluster_to_indices, target_test)
    else:  # subset_sum
        train, test = _assign_subset_sum(cluster_to_indices, y, target_test, n_samples)

    return as_index_array(train), as_index_array(test)


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

    Thin wrapper over ``cluster_split(strategy="centroid")``.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        n_clusters: number of clusters
        random_state: for reproducibility
        **cluster_kwargs: passed to KMeans

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set

    Seed stability: varies with the seed -- built on KMeans clustering, so which
    whole clusters land in test shifts between seeds and the held-out set can
    change substantially.
    """
    return cluster_split(
        embeddings,
        train_size=train_size,
        method="kmeans",
        n_clusters=n_clusters,
        random_state=random_state,
        strategy="centroid",
        **cluster_kwargs,
    )


def cluster_kfold(
    embeddings: ArrayLike,
    y: ArrayLike,
    n_folds: int = 5,
    method: str = "kmeans",
    n_clusters: int | None = None,
    random_state: int = 42,
    **cluster_kwargs: Any,
) -> np.ndarray:
    """Partition data into challenging, label-balanced cross-validation folds.

    Clusters the embeddings, then assigns whole clusters to ``n_folds`` folds so
    that each fold is lexically coherent -- similar samples share a fold, making
    every fold differ from the data trained on for it -- while its label
    distribution stays close to the global one. This yields a "challenging" CV:
    each held-out fold is a cluster-based blind spot rather than a random sample.

    The result is a per-sample fold id, usable directly with
    :class:`sklearn.model_selection.PredefinedSplit`::

        from sklearn.model_selection import PredefinedSplit, cross_validate
        folds = cluster_kfold(embeddings, y, n_folds=5)
        cross_validate(model, X, y, cv=PredefinedSplit(folds))

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        y: class labels of shape (n_samples,); used to balance folds by label
        n_folds: number of cross-validation folds (default 5)
        method: clustering algorithm, 'kmeans' or 'dbscan'
        n_clusters: number of clusters (kmeans only). Defaults to
            ``max(10, n_folds * 3)`` -- comfortably above ``n_folds`` so every
            fold can receive whole clusters.
        random_state: for reproducibility
        **cluster_kwargs: passed to the clustering algorithm

    Returns:
        fold_ids: integer ndarray of shape (n_samples,), values in
        ``[0, n_folds)``. For fold ``k`` the CV test set is ``fold_ids == k``
        and the training set is the rest.

    Raises:
        ValueError: on an unknown ``method``, an ``y``/embeddings length
            mismatch, an out-of-range ``n_folds``, or fewer clusters than folds.

    References:
        Implements the challenging clustering-based cross-validation of Wecker,
        Friedrich & Adel (2020), "ClusterDataSplit: Exploring Challenging
        Clustering-Based Data Splits for Model Performance Evaluation," Eval4NLP
        @ COLING -- folds that are lexically distinct from train while preserving
        label balance. https://aclanthology.org/2020.eval4nlp-1.15

    Seed stability: structure-stable -- fold assignment varies with the KMeans
    clustering, though the cluster-coherent, label-balanced structure is
    preserved.
    """
    embeddings = validate_split_inputs(embeddings, 0.5)  # 0.5: unused placeholder
    y = np.asarray(y)
    n_samples = len(embeddings)

    if len(y) != n_samples:
        raise ValueError(f"y has length {len(y)} but embeddings has {n_samples} rows")
    if not 2 <= n_folds <= n_samples:
        raise ValueError(f"n_folds must be in [2, {n_samples}], got {n_folds}")

    if n_clusters is None:
        n_clusters = max(10, n_folds * 3)
    n_clusters = min(n_clusters, n_samples)
    if n_clusters < n_folds:
        raise ValueError(
            f"need at least n_folds={n_folds} clusters, got n_clusters={n_clusters}"
        )

    if method == "kmeans":
        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto",
            **cluster_kwargs,
        )
    elif method == "dbscan":
        clusterer = DBSCAN(**cluster_kwargs)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    labels = clusterer.fit_predict(embeddings)
    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_to_indices[int(label)].append(idx)

    # Folds are whole clusters, so we need at least n_folds distinct clusters.
    # KMeans is guarded by the n_clusters check above, but DBSCAN chooses its
    # own cluster count and can return fewer (e.g. one dense cluster plus a
    # noise group), which would silently leave CV folds empty. Fail loudly.
    n_found = len(cluster_to_indices)
    if n_found < n_folds:
        raise ValueError(
            f"clustering produced only {n_found} cluster(s), fewer than "
            f"n_folds={n_folds}, which would leave empty folds. Adjust the "
            f"clustering (e.g. DBSCAN eps/min_samples) or reduce n_folds."
        )

    classes = np.unique(y)
    class_idx = {c: k for k, c in enumerate(classes)}

    # Per-cluster class-count vectors.
    clusters = []
    for idxs in cluster_to_indices.values():
        vec = np.zeros(len(classes))
        labs, counts = np.unique(y[idxs], return_counts=True)
        for c, cnt in zip(labs, counts, strict=True):
            vec[class_idx[c]] = cnt
        clusters.append((idxs, vec))

    # Target per-fold class counts: the global label distribution, split n_folds ways.
    global_counts = np.array([np.sum(y == c) for c in classes], dtype=float)
    target = global_counts / n_folds

    # Place the largest clusters first into the fold that overshoots the per-fold
    # target least (so under-filled and empty folds are preferred, keeping labels
    # balanced); ties go to the currently-smallest fold to balance fold sizes.
    fold_counts = [np.zeros(len(classes)) for _ in range(n_folds)]
    fold_ids = np.empty(n_samples, dtype=np.intp)
    clusters.sort(key=lambda t: len(t[0]), reverse=True)
    for idxs, vec in clusters:
        k = min(
            range(n_folds),
            key=lambda f: (
                float(np.maximum(fold_counts[f] + vec - target, 0).sum()),
                float(fold_counts[f].sum()),
            ),
        )
        fold_counts[k] = fold_counts[k] + vec
        fold_ids[idxs] = k

    return fold_ids


# DeepCluster-lite surrogate config (see ``_deepcluster_labels``). Deliberately
# tiny and heavily under-fit, and iterated a few times: a comparison sweep showed
# that a well-fit MLP just reproduces the (label-homogeneous) pseudo-labels, so it
# is the *aggressive* under-fitting + repeated clustering that actually breaks up
# label-pure clusters -- the label diversity we want from a DEEP CLUSTER stand-in.
_DEEPCLUSTER_HIDDEN = (16,)
_DEEPCLUSTER_MAX_ITER = 10
_DEEPCLUSTER_ITERS = 3


def _mlp_hidden_repr(mlp: MLPClassifier, X: np.ndarray) -> np.ndarray:
    """Forward ``X`` through all but the output layer of a fitted ``MLPClassifier``,
    returning the last hidden layer's activations (the learned representation)."""
    activations = {
        "relu": lambda z: np.maximum(z, 0.0),
        "tanh": np.tanh,
        "logistic": lambda z: 1.0 / (1.0 + np.exp(-z)),
        "identity": lambda z: z,
    }
    f = activations[mlp.activation]
    h = np.asarray(X, dtype=float)
    for w, b in zip(mlp.coefs_[:-1], mlp.intercepts_[:-1], strict=True):
        h = f(h @ w + b)
    return h


def _ward_labels(embeddings: np.ndarray, n_clusters: int, **kwargs: Any) -> np.ndarray:
    """Ward's agglomerative clustering -- the paper's base clusterer (deterministic)."""
    n_clusters = min(n_clusters, len(embeddings))
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", **kwargs)
    return clusterer.fit_predict(embeddings)


def _deepcluster_labels(
    embeddings: np.ndarray, n_clusters: int, random_state: int
) -> np.ndarray:
    """A cheap, gradient-based-but-CPU-light surrogate for DEEP CLUSTER (Caron et al.,
    2018), the algorithm Reif & Schwartz (2023) use to obtain *label-diverse* clusters.

    Mirrors one deep-clustering iteration without a heavyweight encoder: Ward-cluster
    the embeddings into pseudo-labels, fit a small under-fit MLP mapping the embeddings
    to those pseudo-labels, take its hidden representation, and re-cluster that. The
    under-fitting (tiny net, few iters) is the diversifying mechanism -- it blends the
    original geometry with the pseudo-structure rather than reproducing the (often
    label-pure) starting clusters. NOT the faithful DEEP CLUSTER (no fine-tuned encoder);
    it is a static-feature stand-in appropriate for a drop-in splitter.
    """
    z = np.asarray(embeddings, dtype=float)
    for _ in range(_DEEPCLUSTER_ITERS):
        pseudo = _ward_labels(z, n_clusters)
        if len(np.unique(pseudo)) < 2:
            break  # a single pseudo-cluster: nothing for the MLP to separate
        mlp = MLPClassifier(
            hidden_layer_sizes=_DEEPCLUSTER_HIDDEN,
            max_iter=_DEEPCLUSTER_MAX_ITER,
            activation="relu",
            random_state=random_state,
        )
        with warnings.catch_warnings():
            # We under-fit on purpose; silence the expected non-convergence warning.
            warnings.simplefilter("ignore")
            mlp.fit(embeddings, pseudo)
        z = _mlp_hidden_repr(mlp, embeddings)
    return _ward_labels(z, n_clusters)


def _minority_cluster_labels(
    embeddings: np.ndarray,
    method: str,
    n_clusters: int,
    random_state: int,
    **cluster_kwargs: Any,
) -> np.ndarray:
    """Cluster ``embeddings`` with the requested method, returning integer labels.

    ``kmeans``/``dbscan`` are the original fast, label-homogeneous clusterers;
    ``ward`` is the paper's deterministic base clusterer; ``deepcluster-lite`` is the
    label-diversifying surrogate (see :func:`_deepcluster_labels`).
    """
    if method == "kmeans":
        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto",
            **cluster_kwargs,
        )
        return clusterer.fit_predict(embeddings)
    if method == "dbscan":
        return DBSCAN(**cluster_kwargs).fit_predict(embeddings)
    if method == "ward":
        return _ward_labels(embeddings, n_clusters, **cluster_kwargs)
    if method == "deepcluster-lite":
        if cluster_kwargs:
            raise ValueError("deepcluster-lite takes no extra cluster_kwargs")
        return _deepcluster_labels(embeddings, n_clusters, random_state)
    raise ValueError(f"Unknown clustering method: {method}")


def minority_route(
    cluster_labels: ArrayLike,
    y: ArrayLike,
    minority_labels: str = "all_but_majority",
) -> tuple[np.ndarray, np.ndarray]:
    """Route a precomputed clustering into a bias-amplified minority split.

    The label-only half of :func:`minority_split`: given an existing cluster
    assignment (from any source -- kmeans, Ward, or an external clusterer such as a
    faithful DEEP CLUSTER pass), send each cluster's *majority*-label instances to
    train and its *minority*-label instances to test. Pure and gradient-free -- it
    inspects only per-cluster label counts. Exposed so heavier, out-of-library
    clusterers can reuse the exact same faithful routing (incl. footnote 10).

    Args:
        cluster_labels: integer cluster id per sample, shape (n_samples,).
        y: class labels, shape (n_samples,).
        minority_labels: ``'all_but_majority'`` (default) or ``'least_only'`` --
            see :func:`minority_split`.

    Returns:
        (train_indices, test_indices), each a sorted index array.

    Raises:
        ValueError: on an unknown ``minority_labels``, a length mismatch, or if no
            minority examples exist (every cluster is label-pure).
    """
    if minority_labels not in ("all_but_majority", "least_only"):
        raise ValueError(f"Unknown minority_labels: {minority_labels}")
    cluster_labels = np.asarray(cluster_labels)
    y = np.asarray(y)
    if len(cluster_labels) != len(y):
        raise ValueError(
            f"cluster_labels has length {len(cluster_labels)} but y has {len(y)}"
        )

    cluster_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        cluster_to_indices[int(label)].append(idx)

    train: list[int] = []
    test: list[int] = []
    for idxs in cluster_to_indices.values():
        idx_arr = np.asarray(idxs)
        cls_labels = y[idx_arr]
        vals, counts = np.unique(cls_labels, return_counts=True)
        if len(vals) < 2:
            train.extend(idx_arr.tolist())  # label-pure cluster: no minority here
            continue
        if minority_labels == "least_only":
            # Footnote 10 (Reif & Schwartz, 2023): only the single least-frequent
            # label per cluster is anti-biased. Keeps the test set small on
            # many-class data, where "all but majority" would flood it.
            minority = vals[np.argmin(counts)]  # ties -> lowest label (deterministic)
            is_test = cls_labels == minority
        else:  # "all_but_majority": every non-majority label is anti-biased
            majority = vals[np.argmax(counts)]  # ties -> lowest label (deterministic)
            is_test = cls_labels != majority
        train.extend(idx_arr[~is_test].tolist())
        test.extend(idx_arr[is_test].tolist())

    if not test:
        raise ValueError(
            "no minority examples found (every cluster is label-pure); "
            "try a different n_clusters or embeddings"
        )
    return as_index_array(sorted(train)), as_index_array(sorted(test))


def minority_split(
    embeddings: ArrayLike,
    y: ArrayLike,
    n_clusters: int = 10,
    method: str = "kmeans",
    random_state: int = 42,
    minority_labels: str = "all_but_majority",
    **cluster_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bias-amplified split via per-cluster minority labels.

    Clusters the embeddings, then for each cluster routes instances whose label
    is the cluster's *majority* label to train (the "biased" majority a
    shortcut-learning model would exploit) and instances with any *minority*
    label to test (the "anti-biased" examples that defy the local pattern),
    yielding a hard, bias-amplified evaluation set.

    Unlike the size-driven splitters, the train/test sizes are determined by the
    data's bias structure (test = the minority-label instances), so this function
    takes no ``train_size``. It inspects per-cluster label counts only; it never
    references global class frequencies.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        y: class labels of shape (n_samples,)
        n_clusters: number of clusters (ignored by 'dbscan')
        method: clustering algorithm. ``'kmeans'`` (default, fast) or ``'dbscan'``
            are standard clusterers that -- as the paper notes -- tend to produce
            *label-homogeneous* clusters, so few minority examples are found and the
            test set can be degenerately small. ``'ward'`` is the paper's
            deterministic base clusterer (agglomerative, Ward linkage; O(n^2) memory).
            ``'deepcluster-lite'`` is a light surrogate for the paper's DEEP CLUSTER
            step that seeks *label-diverse* clusters (hence a larger, non-degenerate
            minority set); see the References note below on faithfulness.
        random_state: for reproducibility (used by 'kmeans' and 'deepcluster-lite')
        minority_labels: which of a cluster's labels are anti-biased (test).
            ``'all_but_majority'`` (default) sends every non-majority label to test;
            ``'least_only'`` (the paper's footnote 10) sends only the single
            least-frequent label. Prefer ``'least_only'`` on many-class data: with
            few clusters relative to the class count, ``'all_but_majority'`` floods
            test (e.g. ~93% on 150-class CLINC with 10 clusters), which the paper
            avoids once minority examples would exceed ~40% of the data.
        **cluster_kwargs: passed to the clustering algorithm

    Returns:
        train_indices: majority-label ("biased") instances, pooled over clusters
        test_indices: minority-label ("anti-biased") instances, pooled over clusters

    Raises:
        ValueError: on an unknown ``method`` or ``minority_labels``, a ``y``/embeddings
            length mismatch, or if no minority examples exist (every cluster is
            label-pure -- try a different ``n_clusters`` or embeddings).

    Warns:
        UserWarning: if the test set is degenerately small (under 5% of the
            samples). The clusters are then nearly label-pure, so the split has
            too few minority examples to be informative -- prefer a different
            ``n_clusters`` or a ``train_size``-driven splitter such as
            :func:`class_boundary_split`.

    References:
        Implements the "minority examples" notion of bias from Reif & Schwartz
        (2023), "Fighting Bias with Bias: Promoting Model Robustness by Amplifying
        Dataset Biases," Findings of ACL, pp. 13169-13189
        (https://aclanthology.org/2023.findings-acl.833): cluster the data, treat
        each cluster's majority label as biased (train) and its minority labels as
        anti-biased (test). Their other two notions -- dataset cartography
        (training dynamics) and partial-input models -- are model-in-the-loop /
        task-specific and out of scope for a static splitter.

        Faithfulness caveat: the paper deliberately does NOT cluster with a standard
        algorithm. It reports that k-means/Ward produce label-homogeneous clusters
        (too few minority examples) and instead uses DEEP CLUSTER (Caron et al.,
        2018) -- a fine-tuned encoder trained on cluster pseudo-labels -- to obtain
        label-diverse clusters. Our default (``method='kmeans'``) is the standard
        clusterer the paper rejects, so it can under-produce minority examples and
        yield a small test set (hence the degenerate-size warning above).
        ``method='ward'`` matches the paper's *base* clusterer (still homogeneous);
        ``method='deepcluster-lite'`` is a static-feature stand-in for the full DEEP
        CLUSTER step (no encoder fine-tuning -- it under-fits a small MLP on
        pseudo-labels and re-clusters its hidden representation) that recovers some
        label diversity at CPU cost.

    Seed stability: with ``'kmeans'``/``'deepcluster-lite'``, nearly deterministic given a
    fixed ``random_state`` (the per-cluster minority labels are stable; only the
    clustering wobbles slightly). ``'ward'`` is fully deterministic (no seed).
    """
    if minority_labels not in ("all_but_majority", "least_only"):
        # Fail before the (possibly expensive, e.g. deepcluster-lite) clustering
        # pass below rather than after it -- minority_route validates this too,
        # but only once clustering has already run.
        raise ValueError(f"Unknown minority_labels: {minority_labels}")
    embeddings = validate_split_inputs(embeddings, 0.5)  # 0.5: unused placeholder
    y = np.asarray(y)
    n_samples = len(embeddings)
    if len(y) != n_samples:
        raise ValueError(f"y has length {len(y)} but embeddings has {n_samples} rows")

    labels = _minority_cluster_labels(
        embeddings, method, n_clusters, random_state, **cluster_kwargs
    )
    train_idx, test_idx = minority_route(labels, y, minority_labels)

    test_fraction = len(test_idx) / n_samples
    if test_fraction < _MINORITY_DEGENERATE_FRACTION:
        warnings.warn(
            f"minority_split produced a degenerate test set: {len(test_idx)} of "
            f"{n_samples} samples ({test_fraction:.1%}). The clusters are nearly "
            "label-pure, so almost no 'minority' examples exist and this split is "
            "likely uninformative. Try a different n_clusters, or a splitter with "
            "an explicit train_size such as class_boundary_split.",
            stacklevel=2,
        )

    return train_idx, test_idx


_STRATIFY_MODES = ("none", "global", "per_class")


def _proportional_quota(y: np.ndarray, target: int) -> dict[Any, int]:
    """Per-class integer test quotas summing to ``target``, proportional to each
    class's frequency in ``y`` (largest-remainder rounding for an exact sum)."""
    classes, counts = np.unique(y, return_counts=True)
    raw = counts / counts.sum() * target
    quota = np.floor(raw).astype(int)
    short = target - int(quota.sum())
    if short:  # hand leftover seats to the largest fractional parts
        quota[np.argsort(-(raw - quota))[:short]] += 1
    return dict(zip(classes.tolist(), quota.tolist(), strict=True))


def _grow_global(
    embeddings: np.ndarray,
    y: np.ndarray,
    test_mask: np.ndarray,
    target_n_test: int,
    metric: str,
    *,
    stratify: bool,
) -> None:
    """Grow ``test_mask`` by single-linkage proximity to the whole current test
    set. If ``stratify``, cap each class at its proportional quota -- a class drops
    out of the shared frontier once full. Mutates ``test_mask`` in place."""
    # min_dist[i] = distance from sample i to its nearest current test member.
    min_dist = cdist(embeddings, embeddings[test_mask], metric=metric).min(axis=1)

    # `blocked[i]` marks samples that can no longer be picked: already in test, or
    # (when stratifying) belonging to a class that has filled its quota.
    if stratify:
        quota = _proportional_quota(y, target_n_test)

        def _full(c: Any) -> bool:
            return int(((y == c) & test_mask).sum()) >= quota[c]

        blocked = test_mask.copy()
        for c in np.unique(y):
            if _full(c):
                blocked |= y == c
    else:
        blocked = test_mask  # proximity-only growth; nothing else to block

    min_dist[blocked] = np.inf
    while test_mask.sum() < target_n_test:
        cand = int(np.argmin(min_dist))  # nearest growable sample to the test set
        if not np.isfinite(min_dist[cand]):
            break  # nothing left to grow into (all classes full / unreachable)
        test_mask[cand] = True
        d_new = cdist(embeddings, embeddings[cand : cand + 1], metric=metric)[:, 0]
        min_dist = np.minimum(min_dist, d_new)
        # Re-mask everything blocked: np.minimum above can revive a previously
        # excluded point (a test member, or a full-class member when stratifying)
        # whose distance to the new pick is small, which would make argmin keep
        # re-picking it and spin forever.
        if stratify:
            if _full(y[cand]):
                blocked |= y == y[cand]
            blocked |= test_mask
            min_dist[blocked] = np.inf
        else:
            min_dist[test_mask] = np.inf


def _grow_per_class(
    embeddings: np.ndarray,
    y: np.ndarray,
    test_mask: np.ndarray,
    target_n_test: int,
    metric: str,
) -> None:
    """Grow each class's own nearest-to-seed neighborhood up to its proportional
    quota, independently of the other classes. Mutates ``test_mask`` in place."""
    quota = _proportional_quota(y, target_n_test)
    seed_idx = np.flatnonzero(test_mask)
    for c, q in quota.items():
        members = np.flatnonzero(y == c)
        if int(test_mask[members].sum()) >= q:
            continue  # seed already fills (or exceeds) this class's quota
        if not test_mask[members].any():
            # No seed member of this class: anchor at the class sample nearest the
            # overall seed, so its region stays near the hard core.
            d_to_seed = cdist(
                embeddings[members], embeddings[seed_idx], metric=metric
            ).min(axis=1)
            test_mask[members[int(np.argmin(d_to_seed))]] = True

        # Single-linkage growth restricted to class c. `in_test` is a bool mask
        # over `members` (local indices); keep it in sync with test_mask.
        in_test = test_mask[members]
        min_dist = cdist(
            embeddings[members], embeddings[members[in_test]], metric=metric
        ).min(axis=1)
        min_dist[in_test] = np.inf
        while int(test_mask[members].sum()) < q:
            local = int(np.argmin(min_dist))
            if not np.isfinite(min_dist[local]):
                break
            test_mask[members[local]] = True
            in_test[local] = True
            d_new = cdist(
                embeddings[members], embeddings[members[local : local + 1]], metric=metric
            )[:, 0]
            min_dist = np.minimum(min_dist, d_new)
            min_dist[in_test] = np.inf  # re-mask in-class members (revival guard)


def _subsample_seed(
    seed: np.ndarray, y: np.ndarray, target: int, random_state: int
) -> np.ndarray:
    """Deterministically shrink an oversized minority seed to exactly ``target``
    indices. Samples class-by-class in round-robin (every class contributes one
    before any contributes a second), so class coverage is preserved as far as the
    budget allows -- the opposite failure mode to growth. Assumes ``target`` is
    strictly smaller than ``len(seed)``."""
    rng = check_random_state(random_state)
    by_class = {
        int(c): rng.permutation(seed[y[seed] == c]).tolist()
        for c in np.unique(y[seed])  # np.unique is sorted -> deterministic order
    }
    kept: list[int] = []
    while len(kept) < target:
        for members in by_class.values():
            if members:
                kept.append(members.pop())
                if len(kept) >= target:
                    break
    return as_index_array(sorted(kept))


def minority_grow_split(
    embeddings: ArrayLike,
    y: ArrayLike,
    train_size: float | int = 0.7,
    n_clusters: int = 10,
    method: str = "kmeans",
    metric: str = "euclidean",
    random_state: int = 42,
    stratify: str = "none",
    minority_labels: str = "all_but_majority",
    **cluster_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Region-grown variant of :func:`minority_split` with a target test size.

    Seeds the test set with the per-cluster minority-label ("anti-biased")
    instances found by :func:`minority_split`, then greedily grows it: at each
    step the not-yet-test sample *closest* to the current test set (single-link
    distance to its nearest test member) is moved to test, until the test set
    reaches the ``train_size``-implied target. This keeps the hard, bias-defying
    flavor of the minority seed while letting you dial the test fraction --
    unlike :func:`minority_split`, whose size is dictated by the data and is
    often degenerately small.

    Class balance is controlled by ``stratify`` (mirrors ``cluster_split``'s
    ``strategy`` string convention):

    - ``"none"`` (default): growth is purely geometric and ignores ``y``, so the
      test set is generally *not* class-balanced -- it expands into one region and
      inherits that region's dominant class (same un-balanced behavior as
      :func:`minority_split`).
    - ``"global"``: one shared proximity frontier, but each class is capped at its
      data-proportional share of the test set; a class drops out of the frontier
      once full. The test set stays one contiguous region whose label mix tracks
      the data.
    - ``"per_class"``: each class grows its *own* nearest-to-seed neighborhood up
      to its quota, independently. The test set is the union of these per-class
      regions (so not a single contiguous blob), but every class contributes its
      geometrically-tightest hard samples. A class with no seed member is anchored
      at its sample nearest the overall seed.

    For both stratified modes the quotas are data-proportional and the minority
    seed is always kept, so a class whose seed already over-fills its quota may
    still exceed its share.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        y: class labels of shape (n_samples,)
        train_size: fraction in (0, 1) or absolute count for the training set;
            sets the target test size (``n_samples - n_train``).
        n_clusters: number of clusters for the minority seed (ignored by 'dbscan')
        method: clustering algorithm for the seed -- see :func:`minority_split`
            ('kmeans', 'dbscan', 'ward', or 'deepcluster-lite')
        metric: distance metric (``scipy.spatial.distance.cdist``) for growing
        random_state: for reproducibility of the clustering seed (and of any
            oversized-seed subsampling)
        stratify: class-balance policy for growth, one of ``"none"`` (default,
            proximity only), ``"global"`` (shared frontier with per-class quotas),
            or ``"per_class"`` (independent per-class neighborhoods). See above.
        minority_labels: which cluster labels seed the test set, ``'all_but_majority'``
            (default) or ``'least_only'`` (footnote 10) -- see :func:`minority_split`.
            Prefer ``'least_only'`` on many-class data; note that either way an
            oversized seed is subsampled back to the target size (below).
        **cluster_kwargs: passed to the seed clustering algorithm

    Returns:
        train_indices: ndarray of indices for the training set
        test_indices: ndarray of indices for the test set (the minority seed plus
            its grown neighborhood). A seed larger than the target is subsampled
            down to it (round-robin across classes to preserve coverage); a seed
            equal to the target is returned as-is. Under a stratified mode the test
            set may still fall short of the target if every class hits its quota
            first.

    Raises:
        ValueError: on an unknown ``method``, ``minority_labels`` or ``stratify``, a
            ``y``/embeddings length mismatch, or if the seed is empty (every cluster
            is label-pure -- see :func:`minority_split`).

    Warns:
        UserWarning: if the resulting test set contains no samples for one or more
            classes (those classes then can't be evaluated). Prefer a stratified
            mode or a larger ``train_size``.

    References:
        Extends the bias-amplified minority seed of Reif & Schwartz (2023); see
        :func:`minority_split`. The proximity growth is the single-linkage region
        growth also used by ``cluster_split(strategy="closest")``.

    Seed stability: nearly deterministic -- the minority seeds and proximity
    growth are largely fixed; only the underlying clustering wobbles between
    seeds. This holds in every ``stratify`` mode (none / global / per-class).
    """
    embeddings = validate_split_inputs(embeddings, train_size)
    y = np.asarray(y)
    n_samples = len(embeddings)
    if len(y) != n_samples:
        raise ValueError(f"y has length {len(y)} but embeddings has {n_samples} rows")
    if stratify not in _STRATIFY_MODES:
        raise ValueError(
            f"Unknown stratify: {stratify!r}. Choose from {list(_STRATIFY_MODES)}"
        )

    # Seed with minority_split; suppress its degenerate-size warning since the
    # whole point here is to grow that seed to a usable size.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, seed = minority_split(
            embeddings,
            y,
            n_clusters=n_clusters,
            method=method,
            minority_labels=minority_labels,
            random_state=random_state,
            **cluster_kwargs,
        )

    target_n_test = n_samples - resolve_n_train(n_samples, train_size)
    target_n_test = max(1, min(target_n_test, n_samples - 1))

    if len(seed) > target_n_test:
        # Grow can only add, never remove, so an oversized seed (e.g. the
        # 'all_but_majority' flood on many-class data) would otherwise blow past
        # train_size and starve train of classes. Shrink it back to target.
        seed = _subsample_seed(seed, y, target_n_test, random_state)

    test_mask = np.zeros(n_samples, dtype=bool)
    test_mask[seed] = True

    if test_mask.sum() < target_n_test:
        if stratify == "per_class":
            _grow_per_class(embeddings, y, test_mask, target_n_test, metric)
        else:
            _grow_global(
                embeddings, y, test_mask, target_n_test, metric,
                stratify=stratify == "global",
            )

    test_idx = np.flatnonzero(test_mask)
    train_idx = np.flatnonzero(~test_mask)

    # A class wholly absent from the test set can't be evaluated. Possible here
    # because the minority seed skips majority-everywhere classes and proximity
    # growth need not reach them (stratify quotas mitigate but don't guarantee
    # when a class's proportional quota rounds to 0).
    missing = np.setdiff1d(np.unique(y), y[test_idx])
    if missing.size:
        warnings.warn(
            f"minority_grow_split produced a test set with no samples for "
            f"class(es) {missing.tolist()}, so those classes cannot be evaluated. "
            "Try stratify='global' or 'per_class', a larger train_size (bigger "
            "test set), or a different n_clusters.",
            stacklevel=2,
        )

    return as_index_array(train_idx), as_index_array(test_idx)


def _other_class_distance(
    embeddings: np.ndarray,
    members: np.ndarray,
    others: np.ndarray,
    y: np.ndarray,
    reference: str,
    metric: str,
) -> np.ndarray:
    """Min distance from each class-``k`` member to a *different* class.

    With ``reference="samples"`` this is the nearest-enemy distance (distance to
    the closest sample of any other class); with ``reference="centroids"`` it is
    the distance to the nearest other-class centroid (cheaper, O(n*K) vs O(n^2)).
    """
    if reference == "samples":
        return cdist(embeddings[members], embeddings[others], metric=metric).min(axis=1)
    # "centroids": one centroid per other class, then nearest among them.
    other_labels = np.unique(y[others])
    centroids = np.vstack(
        [embeddings[others][y[others] == c].mean(axis=0) for c in other_labels]
    )
    return cdist(embeddings[members], centroids, metric=metric).min(axis=1)


def class_boundary_split(
    embeddings: ArrayLike,
    y: ArrayLike,
    train_size: float | int = 0.7,
    metric: str = "euclidean",
    reference: str = "centroids",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Class-stratified adversarial split on per-class decision boundaries.

    Within each class, ranks samples by how close they sit to *other* classes
    and routes the closest (most confusable, near-boundary) ones to the test set
    until that class's test quota is filled. The remaining, more class-typical
    samples go to train. Because the quota is applied per class, the test set
    stays label-balanced (stratified) while still being hard: it concentrates the
    examples a model is most likely to confuse across categories.

    "Closeness to other classes" is measured by ``reference``:

    - ``"centroids"`` (default): distance to the nearest *other* class centroid.
      Cheap -- O(n*K) for K classes.
    - ``"samples"``: nearest-enemy distance, i.e. distance to the single closest
      sample belonging to any other class. Sharper boundaries, but O(n^2).

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        y: class labels of shape (n_samples,)
        train_size: fraction in (0, 1) or absolute count, applied *per class* to
            size each class's train portion. A fraction is recommended; an
            absolute count is clamped to each class's size.
        metric: distance metric passed to ``scipy.spatial.distance.cdist``
        reference: ``"centroids"`` or ``"samples"`` (see above)
        random_state: accepted for API consistency; this split is deterministic

    Returns:
        train_indices: ndarray of class-typical (interior) sample indices
        test_indices: ndarray of near-boundary, cross-class-confusable indices

    Raises:
        ValueError: on a ``y``/embeddings length mismatch, an unknown
            ``reference``, or fewer than two distinct classes (no "other class"
            to measure distance to).

    References:
        A label-stratified, per-class variant of the adversarial-distance idea of
        Søgaard, Ebert, Bastings & Filippova (2021), "We Need to Talk About
        Random Splits," EACL (https://aclanthology.org/2021.eacl-main.156): hard
        splits push dissimilar samples into test. Here "dissimilar" is measured
        toward *other classes* rather than toward the training set, yielding a
        boundary-focused test set; contrast :func:`distance_adversarial_split`
        (unsupervised, distance from the global centroid).

    Seed stability: deterministic -- closeness to other classes is a fixed
    geometric quantity, so the seed has no effect (it is accepted only for API
    consistency).
    """
    embeddings = validate_split_inputs(embeddings, train_size)
    y = np.asarray(y)
    n_samples = len(embeddings)
    if len(y) != n_samples:
        raise ValueError(f"y has length {len(y)} but embeddings has {n_samples} rows")
    if reference not in ("centroids", "samples"):
        raise ValueError(f"reference must be 'centroids' or 'samples', got {reference!r}")

    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(
            f"need at least 2 distinct classes for a boundary split, got {len(classes)}"
        )

    train: list[int] = []
    test: list[int] = []
    for k in classes:
        members = np.flatnonzero(y == k)
        others = np.flatnonzero(y != k)
        n_train_k = min(resolve_n_train(len(members), train_size), len(members))
        n_test_k = len(members) - n_train_k
        if n_test_k <= 0:
            train.extend(members.tolist())
            continue
        dist = _other_class_distance(embeddings, members, others, y, reference, metric)
        # Closest to other classes first -> test; stable sort keeps ties deterministic.
        order = np.argsort(dist, kind="stable")
        test.extend(members[order[:n_test_k]].tolist())
        train.extend(members[order[n_test_k:]].tolist())

    return as_index_array(sorted(train)), as_index_array(sorted(test))


def decision_boundary_split(
    embeddings: ArrayLike,
    y: ArrayLike,
    train_size: float | int = 0.7,
    *,
    model: str = "linear_svc",
    score: str = "confidence",
    stratify: str = "per_class",
    cv: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Supervised adversarial split on a learned decision boundary.

    Fits a fast linear classifier under cross-validation and routes the samples
    a real model finds hardest -- those closest to the learned decision boundary
    -- to the test set, leaving the confident, class-interior samples for train.
    The learned-boundary counterpart to :func:`class_boundary_split` (which
    instead measures geometric distance to other classes).

    Hardness is scored *out of fold* via
    :class:`~sklearn.model_selection.StratifiedKFold`: each sample's margin comes
    from a model that did not train on it, so the score reflects genuine
    difficulty rather than in-sample memorization.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim).
        y: class labels of shape (n_samples,).
        train_size: fraction in (0, 1) or absolute count. With
            ``stratify="per_class"`` it sizes each class's train portion (so the
            test set stays label-balanced); with ``"global"`` it sizes the whole
            train set.
        model: surrogate classifier -- ``"linear_svc"`` (default,
            :class:`~sklearn.svm.LinearSVC`), ``"logistic"``
            (:class:`~sklearn.linear_model.LogisticRegression`), or ``"rbf_svc"``
            (kernel SVM, :class:`~sklearn.svm.SVC` with an RBF kernel and
            ``gamma="scale"``). The linear models are fast; ``"rbf_svc"`` captures
            non-linear boundaries -- and finds genuinely harder splits on
            non-linearly-separable embeddings -- but costs O(n^2)+ per fold, so
            reserve it for smaller n. Each is standardized via a per-fold
            :class:`~sklearn.preprocessing.StandardScaler`.
        score: hardness measure. ``"confidence"`` (default) uses the top-1 minus
            top-2 margin and works for both models; ``"entropy"`` uses predictive
            entropy and requires probabilities (``model="logistic"`` only).
        stratify: ``"per_class"`` (default; per-class test quota -> label-balanced
            test) or ``"global"`` (purest-hard; may unbalance classes).
        cv: number of cross-validation folds, clamped to the smallest class
            count. Each fold supplies the out-of-fold margins for its held-out
            samples.
        random_state: seeds the fold shuffling and the classifier; the split is
            deterministic (ties broken by index).

    Returns:
        train_indices: ndarray of confident, class-interior sample indices.
        test_indices: ndarray of near-boundary, model-confusable indices.

    Raises:
        ValueError: on a ``y``/embeddings length mismatch, fewer than two
            distinct classes, an unknown ``model``/``score``/``stratify``, an
            ``entropy`` score with ``linear_svc``, or a class with fewer than two
            samples (too small to hold out).

    References:
        Uncertainty / margin sampling from active learning (Lewis & Gale, 1994;
        Scheffer et al., 2001; Settles, 2009). Like :func:`class_boundary_split`,
        a label-aware variant of the hard-split motivation of Soegaard et al.
        (2021), "We Need to Talk About Random Splits"
        (https://aclanthology.org/2021.eacl-main.156), but using a *learned*
        boundary rather than embedding geometry.

    Seed stability: structure-stable -- the cross-validation folds depend on the
    seed, so which samples sit nearest the learned boundary (and go to test)
    shifts somewhat between seeds. This holds in both ``stratify`` modes
    (per-class and global wobble about equally).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, LinearSVC

    if model not in ("linear_svc", "logistic", "rbf_svc"):
        raise ValueError(
            f"model must be 'linear_svc', 'logistic', or 'rbf_svc', got {model!r}"
        )
    if score not in ("confidence", "entropy"):
        raise ValueError(f"score must be 'confidence' or 'entropy', got {score!r}")
    if stratify not in ("per_class", "global"):
        raise ValueError(f"stratify must be 'per_class' or 'global', got {stratify!r}")
    if score == "entropy" and model != "logistic":
        raise ValueError("score='entropy' needs predict_proba; use model='logistic'.")

    embeddings = validate_split_inputs(embeddings, train_size)
    y = np.asarray(y)
    n_samples = len(embeddings)
    if len(y) != n_samples:
        raise ValueError(f"y has length {len(y)} but embeddings has {n_samples} rows")

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError(
            f"need at least 2 distinct classes for a boundary split, got {len(classes)}"
        )
    if counts.min() < 2:
        raise ValueError(
            "every class needs >= 2 samples for out-of-fold margins; smallest "
            f"class has {int(counts.min())}."
        )

    n_folds = min(cv, int(counts.min()))

    def make_clf():
        if model == "linear_svc":
            clf = LinearSVC(random_state=random_state)
        elif model == "rbf_svc":
            clf = SVC(kernel="rbf", decision_function_shape="ovr",
                      random_state=random_state)
        else:
            clf = LogisticRegression(max_iter=1000, random_state=random_state)
        return make_pipeline(StandardScaler(), clf)

    # Out-of-fold hardness: higher == closer to the boundary == harder.
    hardness = np.empty(n_samples, dtype=float)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for fit_idx, hold_idx in skf.split(embeddings, y):
        pipe = make_clf()
        pipe.fit(embeddings[fit_idx], y[fit_idx])
        Xh = embeddings[hold_idx]

        if score == "entropy":
            P = pipe.predict_proba(Xh)
            hardness[hold_idx] = -np.sum(P * np.log(P + 1e-12), axis=1)
            continue

        if model == "logistic":
            margins = pipe.predict_proba(Xh)
        else:  # linear_svc or rbf_svc
            D = pipe.decision_function(Xh)
            if D.ndim == 1:  # binary: distance to the single boundary
                hardness[hold_idx] = -np.abs(D)
                continue
            if model == "linear_svc":
                # Normalize each OvR column to a geometric distance by its own
                # ||w_k|| so the cross-class top1 - top2 is comparable. The RBF
                # SVC has no coef_; its OvR decision_function is already
                # vote-aggregated across classes, so it is used as-is.
                D = D / np.linalg.norm(pipe[-1].coef_, axis=1)
            margins = D

        top2 = np.sort(margins, axis=1)[:, -2:]
        hardness[hold_idx] = -(top2[:, 1] - top2[:, 0])

    # Route the hardest samples to test (stable sort -> ties broken by index).
    if stratify == "global":
        n_test = n_samples - resolve_n_train(n_samples, train_size)
        test_idx = np.argsort(-hardness, kind="stable")[:n_test]
        test_set = set(test_idx.tolist())
    else:
        test_set = set()
        for k in classes:
            members = np.flatnonzero(y == k)
            n_train_k = min(resolve_n_train(len(members), train_size), len(members))
            n_test_k = len(members) - n_train_k
            if n_test_k <= 0:
                continue
            order = np.argsort(-hardness[members], kind="stable")
            test_set.update(members[order[:n_test_k]].tolist())

    train = [i for i in range(n_samples) if i not in test_set]
    return as_index_array(train), as_index_array(sorted(test_set))


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

    Seed stability: deterministic -- samples are ranked by distance from the
    centroid, so the seed has no effect (it is accepted only for API
    consistency).
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

    Seed stability: deterministic -- samples are ranked by local density, so the
    seed has no effect.
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

    Seed stability: nearly deterministic -- the outlier ranking is fixed; only
    samples right at the train/test cutoff can change between seeds.
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

    Seed stability: deterministic (spectral method) -- the split follows the
    Fiedler vector from a dense eigendecomposition (no random start vector), whose
    otherwise-arbitrary sign is oriented deterministically. (Without that
    orientation the held-out half would flip with the sign, giving disjoint test
    sets between seeds.)
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
        #
        # Solved with a DENSE partial eigendecomposition (scipy.linalg.eigh,
        # subset_by_index=[0, 1]) rather than sparse ARPACK (eigsh, which='SM').
        # ARPACK aimed at the *small* end of the spectrum converges very slowly
        # and, on larger/messier graphs, effectively hangs — tens of thousands of
        # iterations pegging a core for hours before it finally raises. The
        # distance matrix above is already dense and O(n²), so a dense direct
        # solve costs no extra memory class, always terminates, computes only the
        # two eigenpairs we need, and is fully deterministic (no random ARPACK
        # start vector). (A future sparse kNN-graph build — see the cdist TODO
        # above — should pair with a shift-invert solver to scale past O(n²).)
        L = laplacian(similarities, normed=True)

        try:
            # Two smallest eigenpairs only (subset_by_index is 0-based, inclusive).
            eigenvalues, eigenvectors = dense_eigh(L, subset_by_index=[0, 1])

            # Fiedler vector (2nd eigenvector)
            fiedler = eigenvectors[:, 1]

            # The Fiedler sign is mathematically arbitrary and, with eigsh's
            # random start vector, flips between runs — which would flip which
            # half of the cut becomes test (giving a completely different, seed-
            # dependent held-out set). Orient it deterministically (sign of the
            # largest-magnitude component, as in sklearn's svd_flip) so the split
            # is reproducible.
            if fiedler[np.argmax(np.abs(fiedler))] < 0:
                fiedler = -fiedler

            # Partition by Fiedler vector values
            # Sort and split to achieve desired train_size
            sorted_indices = np.argsort(fiedler)

        except Exception as err:
            # Eigendecomposition failed; fall back to a random split. Warn
            # loudly — silently returning a *random* split from an adversarial
            # splitter would misreport the result's difficulty.
            warnings.warn(
                f"min_cut_split: spectral eigendecomposition failed ({err!r}); "
                "falling back to a random split, which is NOT adversarial. "
                "Try method='stoer_wagner' or adjust similarity_threshold.",
                stacklevel=2,
            )
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

    Seed stability: deterministic -- the normalized-cut partition uses a dense
    eigendecomposition (eigh), which is fixed, so the seed has no effect.
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

    # The Fiedler sign is mathematically arbitrary; eigh's choice is LAPACK-
    # dependent and can differ across platforms/builds, which would flip which
    # half becomes test. Orient it deterministically (sign of the largest-
    # magnitude component, as in sklearn's svd_flip) so the "deterministic"
    # claim above holds portably, not just within one machine.
    if fiedler[np.argmax(np.abs(fiedler))] < 0:
        fiedler = -fiedler

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
    neighborhood.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        leaf_size: BallTree leaf size (higher = slower but less memory)
        random_state: for reproducibility (anchor sampling)

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set

    References:
        Adapted from the adversarial split of Søgaard, Ebert, Bastings &
        Filippova (2021), "We Need to Talk About Random Splits," EACL, which
        constructs hard splits by (approximately) maximizing the Wasserstein
        distance between train and test. https://aclanthology.org/2021.eacl-main.156

    Seed stability: varies with the seed like a random split -- the test pocket
    is grown around a randomly chosen anchor point.
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


def mmd_maximized_split(
    embeddings: ArrayLike,
    train_size: float | int = 0.7,
    n_iterations: int = 500,
    kernel: str = "rbf",
    gamma: float | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Maximize Maximum Mean Discrepancy (MMD) between train and test.

    The adversarial dual of :func:`splytters.mmd_minimized_split`: instead of
    matching the two distributions, it pushes them as far apart as possible in
    kernel/MMD terms, yielding a hard, domain-shifted held-out set. Motivated by
    model selection under domain shift -- a maximally-shifted validation set
    tends to select more robust hyperparameters.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        n_iterations: number of optimization iterations
        kernel: kernel type ('rbf' or 'linear')
        gamma: RBF kernel parameter (default: 1/n_features)
        random_state: for reproducibility

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set

    References:
        Implements the *objective* of Napoli & White (2025), "Clustering-Based
        Validation Splits for Model Selection under Domain Shift," TMLR
        (https://openreview.net/forum?id=Q692C0WtiD): the train/validation split
        should maximize the MMD between the two sets. NOTE: this captures that
        objective via swap optimization (like ``mmd_minimized_split``); it does
        not implement their full method -- a constrained kernel k-means (max-MMD
        is shown equivalent to kernel k-means with k=2) solved by linear
        programming to preserve class/group balance, with convergence guarantees
        and Nyström scaling. For class-balanced clustering splits, see
        ``cluster_split(strategy="subset_sum")`` and :func:`cluster_kfold`.

    Seed stability: varies with the seed like a random split -- the swap
    optimization starts from a random split and many assignments reach a similar
    MMD, so the chosen test set differs run to run.
    """
    embeddings = validate_split_inputs(embeddings, train_size)
    n_dims = embeddings.shape[1]

    _gamma = gamma if gamma is not None else 1.0 / n_dims

    def _rbf_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return np.exp(-_gamma * cdist(X, Y, metric="sqeuclidean"))

    def _linear_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return X @ Y.T

    kernel_fn = _rbf_kernel if kernel == "rbf" else _linear_kernel

    def score_fn(X: np.ndarray, train: list[int], test: list[int]) -> float:
        train_data, test_data = X[train], X[test]
        K_tt = kernel_fn(train_data, train_data)
        K_ss = kernel_fn(test_data, test_data)
        K_ts = kernel_fn(train_data, test_data)
        m, n = len(train_data), len(test_data)
        return (K_tt.sum() / (m * m) +
                K_ss.sum() / (n * n) -
                2 * K_ts.sum() / (m * n))

    # minimize=False -> push the two distributions apart (adversarial dual).
    return optimized_split(
        embeddings, train_size, n_iterations, score_fn, random_state, minimize=False
    )


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

    # KMeans requires n_clusters <= n_samples; clamp so this helper (and its
    # caller split_report) works on small inputs instead of raising.
    n_clusters = max(1, min(n_clusters, len(embeddings)))

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


def maximin_split(
    embeddings: ArrayLike,
    train_size: float | int = 0.7,
    metric: str = "euclidean",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adversarial split via farthest-point (k-center) test selection.

    Builds the test set by greedy farthest-first traversal: seed with one point,
    then repeatedly add the point farthest (``metric``) from everything already
    selected. The result is a maximally *spread-out* test set that covers the
    corners and extremes of the embedding space — a diverse, hard evaluation
    that, unlike a random sample, never under-represents sparse / outlying
    regions. Train is the (denser) remainder.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim).
        train_size: fraction in (0, 1) or absolute count for the training set.
        metric: distance metric passed to ``scipy.spatial.distance.cdist``.
        random_state: seeds the starting point (the traversal is otherwise
            deterministic).

    Returns:
        train_indices: the denser interior remainder.
        test_indices: a farthest-point-sampled, space-covering test set.

    Seed stability: structure-stable -- only the starting point of the
    farthest-first traversal is random, so different seeds give overlapping but
    not identical space-covering test sets.
    """
    embeddings = validate_split_inputs(embeddings, train_size)
    n_samples = len(embeddings)
    n_test = n_samples - resolve_n_train(n_samples, train_size)
    rng = check_random_state(random_state)

    # TODO: Replace the full pairwise matrix with incremental NearestNeighbors
    # queries to avoid materializing O(n²) distances.
    distances = cdist(embeddings, embeddings, metric=metric)

    start = int(rng.randint(n_samples))
    selected = [start]
    selected_mask = np.zeros(n_samples, dtype=bool)
    selected_mask[start] = True
    # Min distance from each point to the current test set; -1 marks selected.
    min_dist = distances[start].copy()
    min_dist[selected_mask] = -1.0
    while len(selected) < n_test:
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        selected_mask[nxt] = True
        min_dist = np.minimum(min_dist, distances[nxt])
        min_dist[selected_mask] = -1.0

    test_set = set(selected)
    train = [i for i in range(n_samples) if i not in test_set]
    return as_index_array(train), as_index_array(sorted(test_set))
