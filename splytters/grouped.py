"""
Grouping-aware splits that keep related samples on the same side.

These prevent train/test leakage from samples that must not be separated:
explicit groups (same user / document / source) for :func:`group_split`, and
discovered near-duplicates for :func:`deduplicated_split`. Unlike
:func:`splytters.cluster_kfold`, the groups here are *given* (or derived from a
similarity threshold), not discovered by clustering.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from sklearn.utils import check_random_state

from splytters.utils import as_index_array, resolve_n_train, validate_split_inputs


def _assign_whole_groups(
    group_to_indices: dict, target_train: int, rng
) -> tuple[list[int], list[int]]:
    """Best-fit assign whole groups to train to approach ``target_train`` by
    sample count, rest to test, so no group is split across sides. Guarantees
    both sides non-empty.
    """
    groups = [np.asarray(idxs) for idxs in group_to_indices.values()]
    order = rng.permutation(len(groups))

    train_groups: list[int] = []
    test_groups: list[int] = []
    filled = 0
    for j in order:
        size = len(groups[j])
        # Best-fit: keep the group in train only if doing so lands the running
        # train count strictly closer to the target than leaving it out. A group
        # larger than the whole target is still admitted when it gets us closer,
        # instead of being permanently barred (first-fit grossly undershot
        # train_size whenever a group exceeded the target).
        if abs(filled + size - target_train) < abs(filled - target_train):
            train_groups.append(j)
            filled += size
        else:
            test_groups.append(j)

    # Both sides must stay non-empty. If best-fit swept every group to one side
    # (e.g. a single dominant group, or a target that swallows them all), move
    # the smallest group across; whole groups remain intact either way.
    if not train_groups:
        smallest = min(test_groups, key=lambda j: len(groups[j]))
        test_groups.remove(smallest)
        train_groups.append(smallest)
    elif not test_groups:
        smallest = min(train_groups, key=lambda j: len(groups[j]))
        train_groups.remove(smallest)
        test_groups.append(smallest)

    train = [i for j in train_groups for i in groups[j].tolist()]
    test = [i for j in test_groups for i in groups[j].tolist()]
    return train, test


def group_split(
    embeddings: ArrayLike,
    groups: ArrayLike,
    train_size: float | int = 0.7,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split so that every group lands entirely on one side (no group leakage).

    All samples sharing a group id (e.g. the same user, document, patient, or
    source) are kept together in either train or test. Whole groups are assigned
    best-fit to approach ``train_size`` by sample count (a group larger than the
    target can still land in train when that gets the count closer). The analogue of
    scikit-learn's :class:`~sklearn.model_selection.GroupShuffleSplit`, but on
    embeddings and returning index arrays.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim) (used for
            length/validation; the split is driven by ``groups``).
        groups: group id per sample, shape (n_samples,).
        train_size: fraction in (0, 1) or absolute count for the training set.
            Approximate, since groups are indivisible.
        random_state: for reproducibility.

    Returns:
        train_indices, test_indices.

    Raises:
        ValueError: on a ``groups``/embeddings length mismatch or fewer than two
            distinct groups (nothing to split).

    Seed stability: varies with the seed like a random split -- whole groups are
    assigned to train/test in a random (seeded) order, though each group always
    stays intact.
    """
    embeddings = validate_split_inputs(embeddings, train_size)
    groups = np.asarray(groups)
    n_samples = len(embeddings)
    if len(groups) != n_samples:
        raise ValueError(
            f"groups has length {len(groups)} but embeddings has {n_samples} rows"
        )
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError(f"need at least 2 distinct groups to split, got {len(unique)}")

    rng = check_random_state(random_state)
    group_to_indices = {g: np.flatnonzero(groups == g) for g in unique}
    target_train = resolve_n_train(n_samples, train_size)
    train, test = _assign_whole_groups(group_to_indices, target_train, rng)
    return as_index_array(sorted(train)), as_index_array(sorted(test))


def deduplicated_split(
    embeddings: ArrayLike,
    train_size: float | int = 0.7,
    similarity_threshold: float | None = None,
    metric: str = "euclidean",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split so that near-duplicates never straddle train and test.

    Groups of near-duplicate samples (connected components of the graph linking
    pairs within ``similarity_threshold``) are each assigned entirely to one
    side, preventing the inflated scores that train/test duplicate leakage
    causes. The inverse of :func:`splytters.duplicate_spread_split`, which
    intentionally puts duplicates on *both* sides.

    Args:
        embeddings: array-like of shape (n_samples, embedding_dim).
        train_size: fraction in (0, 1) or absolute count for the training set
            (approximate, since duplicate groups are indivisible).
        similarity_threshold: pairs closer than this (``metric`` distance) are
            treated as near-duplicates. Defaults to the 1st percentile of
            pairwise distances (conservative — only the closest pairs); raise it
            to merge looser near-duplicates, lower it to merge only exact ones.
        metric: distance metric passed to ``scipy.spatial.distance.cdist``.
        random_state: for reproducibility.

    Returns:
        train_indices, test_indices.

    Raises:
        ValueError: if the threshold merges every sample into one near-duplicate
            component (nothing left to split without leakage) — lower
            ``similarity_threshold``.

    Seed stability: structure-stable -- the near-duplicate groups are fixed; only
    which group goes to which side is random.
    """
    embeddings = validate_split_inputs(embeddings, train_size)
    n_samples = len(embeddings)
    rng = check_random_state(random_state)

    # TODO: Replace the full pairwise matrix with BallTree.query_radius to find
    # near-duplicate pairs without materializing O(n²) distances.
    distances = cdist(embeddings, embeddings, metric=metric)
    np.fill_diagonal(distances, np.inf)
    if similarity_threshold is None:
        finite = distances[distances < np.inf]
        similarity_threshold = np.percentile(finite, 1)

    adjacency = (distances <= similarity_threshold).astype(int)
    _, labels = connected_components(csr_matrix(adjacency))

    component_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        component_to_indices[int(label)].append(idx)
    if len(component_to_indices) < 2:
        raise ValueError(
            "similarity_threshold merged all samples into one near-duplicate "
            "component; lower it to leave separable groups."
        )

    components = {k: np.asarray(v) for k, v in component_to_indices.items()}
    target_train = resolve_n_train(n_samples, train_size)
    train, test = _assign_whole_groups(components, target_train, rng)
    return as_index_array(sorted(train)), as_index_array(sorted(test))
