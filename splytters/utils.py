"""
Shared utilities for splitting algorithms.
"""

from __future__ import annotations

import inspect
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.utils import check_array, check_random_state


def accepts_random_state(fn: Any) -> bool:
    """Whether ``fn`` takes a ``random_state`` keyword (directly or via **kwargs).

    Used to decide whether to forward ``random_state`` to a user-supplied
    splitter: passing it to one that doesn't accept it raises ``TypeError``.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return True  # can't introspect (e.g. a C callable) — assume it does
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return True
    return "random_state" in params


def to_numpy(X: ArrayLike) -> Any:
    """Best-effort conversion of framework tensors to numpy.

    Handles PyTorch tensors (including those on GPU / requiring grad) by
    detaching, moving to CPU, and converting. Any other input is returned
    unchanged so the caller's downstream ``check_array``/``np.asarray`` can
    handle lists, numpy arrays, pandas DataFrames, etc.
    """
    # Duck-type torch.Tensor without importing torch.
    if hasattr(X, "detach") and hasattr(X, "cpu") and hasattr(X, "numpy"):
        try:
            return X.detach().cpu().numpy()
        except Exception:  # pragma: no cover - fall through to generic handling
            pass
    return X


def validate_split_inputs(
    embeddings: ArrayLike, train_size: float | int, min_samples: int = 2
) -> np.ndarray:
    """Validate and coerce inputs shared by every splitting function.

    Accepts any array-like (numpy, list, pandas, torch tensor), converts it to
    a finite 2-D float ``ndarray``, and validates ``train_size``.

    Args:
        embeddings: array-like of shape (n_samples, n_features).
        train_size: fraction in the open interval (0, 1), or an absolute count
            in ``[1, n_samples)``. Mirrors
            ``sklearn.model_selection.train_test_split``.
        min_samples: minimum number of samples required to form a split
            (default 2).

    Returns:
        The validated, finite, float embeddings as an ndarray of shape
        (n_samples, n_features).

    Raises:
        ValueError: if ``train_size`` is out of range, there are too few
            samples, or the embeddings contain NaN/inf or are not 2-D.
    """
    # Validate train_size first so its message takes priority (matches the
    # historical "between 0 and 1" contract for fractional sizes).
    is_int = isinstance(train_size, (int, np.integer)) and not isinstance(
        train_size, bool
    )
    if not is_int:
        if not (isinstance(train_size, (float, np.floating)) and 0 < train_size < 1):
            raise ValueError(
                "train_size as a fraction must be between 0 and 1 exclusive, "
                f"got {train_size!r}"
            )

    X = to_numpy(embeddings)
    n = len(X)
    if n < min_samples:
        raise ValueError(f"Need at least {min_samples} samples to split, got {n}")

    if is_int and not (1 <= train_size < n):
        raise ValueError(
            f"train_size as an absolute count must be in [1, {n}), got {train_size}"
        )

    # Coerce to a finite 2-D numeric array (rejects NaN/inf, 1-D, ragged, sparse).
    X = check_array(X, ensure_2d=True, allow_nd=False, dtype="numeric")
    return X


def resolve_n_train(n_samples: int, train_size: float | int) -> int:
    """Resolve ``train_size`` (fraction or absolute count) to an int count.

    A fractional ``train_size`` is clamped to ``[1, n_samples - 1]`` so the
    resolved count never collapses one side of the split to empty (e.g.
    ``n_samples=2, train_size=0.3`` would otherwise truncate to 0).
    """
    if isinstance(train_size, (int, np.integer)) and not isinstance(train_size, bool):
        return int(train_size)
    n_train = int(n_samples * train_size)
    return min(max(n_train, 1), n_samples - 1)


def apportion_train(bin_sizes: list[int], n_train_total: int) -> np.ndarray:
    """Split ``n_train_total`` train slots across bins via largest-remainder.

    Each bin gets ``floor(size * n_train_total / total)`` slots, then the
    leftover slots go to the bins with the largest fractional remainders. The
    per-bin counts sum exactly to ``n_train_total`` (clamped to ``[0, total]``),
    avoiding both the per-bin ``int()`` truncation bias that undershoots the
    requested train fraction and the all-to-train rounding that can empty the
    test set.

    Args:
        bin_sizes: number of samples in each (non-empty) bin.
        n_train_total: total samples to assign to train across all bins.

    Returns:
        Integer ndarray of per-bin train counts, aligned with ``bin_sizes``.
    """
    sizes = np.asarray(bin_sizes, dtype=np.intp)
    total = int(sizes.sum())
    if total == 0:
        return np.zeros(len(sizes), dtype=np.intp)
    n_train_total = int(min(max(n_train_total, 0), total))

    ideal = sizes * (n_train_total / total)
    counts = np.floor(ideal).astype(np.intp)
    remainder = n_train_total - int(counts.sum())
    if remainder > 0:
        # Hand leftover slots to the largest fractional remainders that still
        # have spare capacity (ideal <= size, so capacity always exists here).
        order = np.argsort(-(ideal - np.floor(ideal)))
        for i in order:
            if remainder == 0:
                break
            if counts[i] < sizes[i]:
                counts[i] += 1
                remainder -= 1
    return counts


def as_index_array(indices: ArrayLike) -> np.ndarray:
    """Return ``indices`` as a 1-D integer ndarray (the canonical split output)."""
    return np.asarray(list(indices), dtype=np.intp)


def compute_pairwise_distances(X: ArrayLike, metric: str = "euclidean") -> np.ndarray:
    """Compute pairwise distance matrix.

    .. warning::
        Materializes a full O(n²) matrix. For large datasets, prefer
        NearestNeighbors or chunked computation.
    """
    # TODO: Add an optional max_samples guard or return a sparse representation
    # for large inputs to avoid OOM.
    X = np.asarray(X)
    return cdist(X, X, metric=metric)


def compute_centroid(X: ArrayLike) -> np.ndarray:
    """Compute centroid of embeddings."""
    X = np.asarray(X)
    return X.mean(axis=0)


def compute_split_centroids(
    X: ArrayLike, train_indices: ArrayLike, test_indices: ArrayLike
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute centroids of train and test sets."""
    X = np.asarray(X)
    train_indices = np.asarray(train_indices, dtype=np.intp)
    test_indices = np.asarray(test_indices, dtype=np.intp)
    train_centroid = X[train_indices].mean(axis=0) if len(train_indices) else None
    test_centroid = X[test_indices].mean(axis=0) if len(test_indices) else None
    return train_centroid, test_centroid


def cluster_embeddings(
    X: ArrayLike,
    n_clusters: int = 10,
    method: str = "kmeans",
    random_state: int = 42,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[int, list[int]], np.ndarray]:
    """
    Cluster embeddings and return labels and cluster info.

    Returns:
        labels: cluster label for each sample
        cluster_to_indices: dict mapping cluster_id to list of indices
        cluster_centers: cluster centroids (for kmeans)
    """
    X = np.asarray(X)

    if method == "kmeans":
        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto",
            **kwargs
        )
        labels = clusterer.fit_predict(X)
        cluster_centers = clusterer.cluster_centers_
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    cluster_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_to_indices[label].append(idx)

    return labels, cluster_to_indices, cluster_centers


def random_split(
    embeddings: ArrayLike, train_size: float | int = 0.7, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple random train/test split (baseline).

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim)
        train_size: fraction in (0, 1) or absolute count for the training set
        random_state: int, RandomState, or None for reproducibility

    Returns:
        train_indices: ndarray of indices for training set
        test_indices: ndarray of indices for test set

    Seed stability: fully random -- every seed gives a different train/test
    split; this is the baseline the other splitters' seed stability is measured
    against.
    """
    embeddings = validate_split_inputs(embeddings, train_size)
    n_samples = len(embeddings)
    n_train = resolve_n_train(n_samples, train_size)
    rng = check_random_state(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    return indices[:n_train], indices[n_train:]


def compute_split_similarity(
    X: ArrayLike,
    train_indices: ArrayLike,
    test_indices: ArrayLike,
    metric: str = "euclidean",
) -> dict[str, float]:
    """
    Compute similarity metrics between train and test splits.

    Returns dict with:
        - centroid_distance: distance between train/test centroids
        - mean_cross_distance: mean distance from test to nearest train
        - coverage: fraction of test samples with train neighbor within median distance
    """
    X = np.asarray(X)
    train_indices = np.asarray(train_indices, dtype=np.intp)
    test_indices = np.asarray(test_indices, dtype=np.intp)

    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError(
            "compute_split_similarity requires non-empty train and test sets"
        )

    train_X = X[train_indices]
    test_X = X[test_indices]

    # Centroid distance
    train_centroid = train_X.mean(axis=0)
    test_centroid = test_X.mean(axis=0)
    centroid_distance = np.linalg.norm(train_centroid - test_centroid)

    # Cross-set distances
    cross_distances = cdist(test_X, train_X, metric=metric)
    min_distances = cross_distances.min(axis=1)
    mean_cross_distance = min_distances.mean()

    # Coverage (fraction of test with nearby train sample)
    # TODO: Replace full pairwise matrix with sampled median estimation and
    # NearestNeighbors for coverage check to reduce O(n²) memory.
    all_distances = cdist(X, X, metric=metric)
    np.fill_diagonal(all_distances, np.inf)
    median_dist = np.median(all_distances[all_distances < np.inf])
    coverage = (min_distances <= median_dist).mean()

    return {
        "centroid_distance": float(centroid_distance),
        "mean_cross_distance": float(mean_cross_distance),
        "coverage": float(coverage),
    }


def optimized_split(
    embeddings: np.ndarray,
    train_size: float | int,
    n_iterations: int,
    score_fn: Callable[[np.ndarray, list[int], list[int]], float],
    random_state: int = 42,
    minimize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterative swap-optimization to find a split that optimizes a score.

    Starts from a random split and repeatedly swaps one train/test pair,
    keeping the swap only if the score improves.

    Args:
        embeddings: array of shape (n_samples, embedding_dim), already np.ndarray
        train_size: fraction in (0, 1) or absolute count for the training set
        n_iterations: number of swap attempts
        score_fn: callable(embeddings, train_indices, test_indices) -> float
        random_state: int, RandomState, or None for reproducibility
        minimize: if True, accept swaps that lower the score;
                  if False, accept swaps that raise it
    """
    n_samples = len(embeddings)
    rng = check_random_state(random_state)

    all_indices = np.arange(n_samples)
    rng.shuffle(all_indices)

    n_train = resolve_n_train(n_samples, train_size)
    train_indices = set(all_indices[:n_train].tolist())
    test_indices = set(all_indices[n_train:].tolist())

    current_score = score_fn(embeddings, list(train_indices), list(test_indices))

    for _ in range(n_iterations):
        train_sample = rng.choice(list(train_indices))
        test_sample = rng.choice(list(test_indices))

        # swap
        train_indices.remove(train_sample)
        train_indices.add(test_sample)
        test_indices.remove(test_sample)
        test_indices.add(train_sample)

        new_score = score_fn(embeddings, list(train_indices), list(test_indices))

        improved = new_score < current_score if minimize else new_score > current_score
        if improved:
            current_score = new_score
        else:
            # revert
            train_indices.remove(test_sample)
            train_indices.add(train_sample)
            test_indices.remove(train_sample)
            test_indices.add(test_sample)

    return as_index_array(sorted(train_indices)), as_index_array(sorted(test_indices))


def constrained_kernel_kmeans_split(
    embeddings: np.ndarray,
    train_size: float | int,
    kernel: str = "rbf",
    gamma: float | None = None,
    y: ArrayLike | None = None,
    groups: ArrayLike | None = None,
    random_state: int = 42,
    n_iterations: int = 100,
    maximize_mmd: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Constrained kernel k-means (k=2) train/validation split via an LP.

    Faithful implementation of the partitioning method of Napoli & White
    (TMLR 2025; arXiv 2024), "Clustering-Based Validation Splits for Model
    Selection under Domain Shift" (https://openreview.net/forum?id=Q692C0WtiD).

    The paper (their Theorem 1, Eq. 8-11) proves that maximizing the MMD between
    the two sets is equivalent to *minimizing* the kernel k-means objective
    ``Psi(T, V) = SSq(T) + SSq(V)`` with ``k = 2`` (the within-cluster kernel
    scatter), because ``MMD^2 = c * (SSq(S) - Psi(T, V))`` with ``SSq(S)`` and the
    cluster sizes held constant. It solves the constrained clustering with a
    Lloyd-style alternation (their Algorithm 1):

    1. **Distance update** -- kernel distance from every point ``i`` to each
       cluster centroid ``j`` via the kernel trick,
       ``D_ij = K_ii - (2 / s_j) * sum_l U_lj K_il + (1 / s_j^2) * sum_lm U_lj U_mj K_lm``
       where ``s_j = sum_l U_lj`` is the cluster mass.
    2. **Constrained assignment** -- solve the assignment LP (their Eq. 12-16)
       ``argmin_U sum_ij U_ij D_ij`` subject to ``sum_j U_ij = 1`` (Eq. 14),
       the group-balance equalities ``sum_{i in g} U_ij = round(h |g|)`` for every
       group ``g`` and cluster ``j`` (Eq. 15), and the relaxed box constraint
       ``0 <= U_ij <= 1`` (Eq. 16). The paper drops the integrality constraint
       (Eq. 13) because the constraint matrix meets Hoffman's total-unimodularity
       conditions, so the LP has integral optima.

    Groups ``g`` are the Cartesian product of the label ``y`` and ``groups``
    inputs (their ``Y x D``); with neither, there is a single group and Eq. 15
    reduces to the size constraint alone. The holdout fraction ``h`` is derived
    from ``train_size``; per-group targets are apportioned with largest-remainder
    so they sum exactly to the requested validation size.

    Rounding: even though the LP optima are (near-)integral, fractional values can
    appear under solver degeneracy, so each iteration rounds by selecting, within
    every group, the ``round(h |g|)`` points with the largest validation weight
    ``U[:, 1]``. This guarantees an exact, feasible hard partition.

    Convergence (their Proposition 2): ``Psi`` is bounded below and monotone under
    the alternation and there are finitely many partitions, so the max-MMD
    iteration (full-step assignment) is run until the hard assignment stops
    changing (capped at ``n_iterations``), keeping the best partition seen.

    The min-MMD case (``maximize_mmd=False``) is the library's dual, not part of
    the paper. ``MMD^2`` is convex in the relaxed assignment, so *minimizing* it
    is a convex program: a full assignment step oscillates between the two
    segregated vertices, so this uses a damped Frank-Wolfe update
    ``v <- (1 - gamma) v + gamma * u*`` with ``gamma = 2 / (k + 2)`` (``u*`` the
    LP vertex), then rounds the fractional membership at the end. The rounded
    result lands far below a random split's MMD but, in practice, typically
    around 2x the MMD reached by the swap optimizer -- prefer the swap method
    when the absolute lowest MMD matters and this path when the label/group
    constraints do.

    NOTE: the paper's Nyström scaling (Algorithm 1, Step 1; an ``O(qn)`` random
    submatrix approximation of the kernel for very large ``n``) is not implemented
    here -- this helper materializes the full ``n x n`` kernel and is intended for
    moderate ``n``.

    Args:
        embeddings: validated float ndarray of shape (n_samples, n_features).
        train_size: fraction in (0, 1) or absolute count for the training set.
        kernel: 'rbf' or 'linear'.
        gamma: RBF kernel parameter (default: 1 / n_features).
        y: optional labels; enforces per-label proportions in both sides.
        groups: optional group ids; enforces per-group proportions in both sides.
        random_state: seeds the initial random (feasible) partition.
        n_iterations: maximum Lloyd-style iterations.
        maximize_mmd: if True, minimize the kernel k-means scatter (max-MMD, the
            paper's objective); if False, maximize the scatter (the min-MMD dual,
            an anti-clustering that makes the two sets resemble each other).

    Returns:
        (train_indices, test_indices) as sorted 1-D integer ndarrays.
    """
    from scipy import sparse
    from scipy.optimize import linprog

    n_samples = len(embeddings)
    n_train = resolve_n_train(n_samples, train_size)

    if kernel == "rbf":
        _gamma = gamma if gamma is not None else 1.0 / embeddings.shape[1]
        K = np.exp(-_gamma * cdist(embeddings, embeddings, metric="sqeuclidean"))
    elif kernel == "linear":
        K = embeddings @ embeddings.T
    else:
        raise ValueError(f"kernel must be 'rbf' or 'linear', got {kernel!r}")
    K_diag = np.diag(K).copy()

    # Build group ids as the Cartesian product of y and groups (paper: Y x D).
    keys: list[Any] = []
    for i in range(n_samples):
        key = []
        if y is not None:
            key.append(y[i])
        if groups is not None:
            key.append(groups[i])
        keys.append(tuple(key))
    unique_keys = sorted(set(keys), key=lambda k: repr(k))
    group_members = [
        np.array([i for i in range(n_samples) if keys[i] == k], dtype=np.intp)
        for k in unique_keys
    ]
    group_sizes = [len(m) for m in group_members]

    # Per-group train targets (largest-remainder) -> validation targets. This
    # preserves each group's label/group proportion at the global train fraction
    # and makes the totals sum exactly to n_train / n_val.
    per_group_train = apportion_train(group_sizes, n_train)
    val_targets = [
        sz - tr for sz, tr in zip(group_sizes, per_group_train, strict=True)
    ]

    # Initial feasible partition: pick val_targets[g] members of each group at
    # random for the validation cluster. ``v`` is the (fractional) validation
    # membership weight per point; train membership is ``1 - v``.
    rng = check_random_state(random_state)
    v = np.zeros(n_samples)
    for members, vt in zip(group_members, val_targets, strict=True):
        if vt > 0:
            chosen = rng.choice(members, size=vt, replace=False)
            v[chosen] = 1.0

    # Precompute the equality-constraint system for the assignment LP. Variables
    # are ordered u[2*i + j] for point i, cluster j.
    n_vars = 2 * n_samples
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b_eq: list[float] = []
    # Eq. 14: each point assigned once.
    for i in range(n_samples):
        rows += [i, i]
        cols += [2 * i, 2 * i + 1]
        data += [1.0, 1.0]
        b_eq.append(1.0)
    # Eq. 15: per-group validation mass fixed (train mass then follows from
    # Eq. 14, so only the validation-cluster equality is needed per group).
    r = n_samples
    for members, vt in zip(group_members, val_targets, strict=True):
        for i in members:
            rows.append(r)
            cols.append(2 * i + 1)
            data.append(1.0)
        b_eq.append(float(vt))
        r += 1
    A_eq = sparse.csr_matrix(
        (data, (rows, cols)), shape=(n_samples + len(unique_keys), n_vars)
    )
    b_eq_arr = np.asarray(b_eq)

    def round_membership(weights: np.ndarray) -> set[int]:
        """Round fractional weights to an exact, feasible validation set: within
        each group, take the ``val_targets[g]`` points with the largest weight."""
        chosen_all: set[int] = set()
        for members, vt in zip(group_members, val_targets, strict=True):
            if vt <= 0:
                continue
            order = np.argsort(-weights[members], kind="stable")
            chosen_all.update(int(i) for i in members[order[:vt]])
        return chosen_all

    def mmd_sq(val_set: set[int]) -> float:
        """True (biased) MMD^2 between the current train/validation partition."""
        val = np.zeros(n_samples)
        if val_set:
            val[np.fromiter(val_set, dtype=np.intp, count=len(val_set))] = 1.0
        trn = 1.0 - val
        s_v, s_t = val.sum(), trn.sum()
        Kv, Kt = K @ val, K @ trn
        return float(
            (trn @ Kt) / (s_t * s_t)
            + (val @ Kv) / (s_v * s_v)
            - 2.0 * (trn @ Kv) / (s_t * s_v)
        )

    # Lloyd-style alternation. For max-MMD this is exactly constrained kernel
    # k-means (full-step assignment). MMD^2 is convex in the (relaxed) assignment,
    # so a full step maximizes it toward a vertex, but *minimizing* it (the
    # min-MMD dual, an anti-clustering) needs a damped Frank-Wolfe step
    # ``gamma = 2 / (k + 2)`` to avoid the two-vertex oscillation a full step
    # produces. Track the best hard partition seen and return it.
    best_val: set[int] = round_membership(v)
    best_obj = mmd_sq(best_val)
    prev_val: set[int] | None = None
    for k in range(max(1, n_iterations)):
        # Step 1: kernel distances to the current soft centroids.
        D = np.empty((n_samples, 2))
        for j, w in enumerate((1.0 - v, v)):
            s = w.sum()
            if s <= 0:
                D[:, j] = K_diag  # empty cluster: distance is just K_ii
                continue
            Kw = K @ w
            D[:, j] = K_diag - 2.0 * Kw / s + float(w @ Kw) / (s * s)

        # Step 2: constrained assignment LP. Minimize scatter for max-MMD;
        # maximize it (negate objective) for the min-MMD dual.
        c = D.reshape(-1).copy()
        if not maximize_mmd:
            c = -c
        res = linprog(
            c, A_eq=A_eq, b_eq=b_eq_arr, bounds=(0.0, 1.0), method="highs"
        )
        if not res.success:  # pragma: no cover - LP is always feasible here
            raise RuntimeError(f"assignment LP failed: {res.message}")
        vertex = res.x.reshape(n_samples, 2)[:, 1]

        gamma = 1.0 if maximize_mmd else 2.0 / (k + 2)
        v = (1.0 - gamma) * v + gamma * vertex

        val_set = round_membership(v)
        obj = mmd_sq(val_set)
        improved = obj > best_obj if maximize_mmd else obj < best_obj
        if improved:
            best_obj, best_val = obj, val_set
        # Full-step kernel k-means converges to a stable hard assignment; stop
        # early there. Damped Frank-Wolfe keeps moving, so run the full budget.
        if maximize_mmd and val_set == prev_val:
            break
        prev_val = val_set

    test_indices = sorted(best_val)
    train_indices = [i for i in range(n_samples) if i not in best_val]
    return as_index_array(train_indices), as_index_array(test_indices)


def greedy_assign_to_target(
    items_with_sizes: list[tuple[int, int]], target_size: int
) -> tuple[list[int], list[int]]:
    """
    Greedily assign items to reach target size.

    Args:
        items_with_sizes: list of (item_id, size) tuples
        target_size: target total size

    Returns:
        selected: list of item_ids assigned
        remaining: list of item_ids not assigned
    """
    selected = []
    remaining = []
    current_size = 0

    for item_id, size in items_with_sizes:
        if current_size + size <= target_size:
            selected.append(item_id)
            current_size += size
        else:
            remaining.append(item_id)

    return selected, remaining
