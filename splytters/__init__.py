"""
Splitting algorithms for dataset partitioning.

This package provides functions to create train-test splits with
different objectives:

Modules:
    adversarial: Minimize train-test similarity (hard evaluation)
    overlap: Maximize train-test similarity (easy evaluation, sanity checks)
    balanced: Match distributions between train/test (fair evaluation)
    utils: Shared utilities
"""

# Public type alias for custom splitters (see SplytterSplit / train_test_split).
from splytters._types import Splitter
from splytters.adversarial import (
    centroid_adversarial_split,
    class_boundary_split,
    cluster_kfold,
    cluster_split,
    decision_boundary_split,
    density_adversarial_split,
    distance_adversarial_split,
    get_cluster_info,
    maximin_split,
    min_cut_split,
    minority_grow_split,
    minority_route,
    minority_split,
    mmd_maximized_split,
    normalized_cut_split,
    outlier_adversarial_split,
    wasserstein_adversarial_split,
)
from splytters.balanced import (
    density_balanced_split,
    distribution_matched_split,
    histogram_matched_split,
    mmd_minimized_split,
    moment_matched_split,
    stratified_random_split,
)

# Curriculum / ordering-driven split (pair with a splytters.sorters ranking).
from splytters.curriculum import sorted_stratified_split

# Embedder discovery (heavy model libs stay lazy; listing needs no extra).
from splytters.embedders import list_embedders

# Grouping-aware splits (keep related samples / near-duplicates on one side).
from splytters.grouped import deduplicated_split, group_split

# Framework interop (pandas / torch / HuggingFace datasets). Heavy deps are
# imported lazily inside each helper, so this import stays dependency-light.
from splytters.interop import (
    split_dataframe,
    split_dataset,
    to_torch_subsets,
)
from splytters.overlap import (
    centroid_matched_split,
    cluster_leak_split,
    duplicate_spread_split,
    max_coverage_split,
    nearest_neighbor_split,
    neighbor_coverage_split,
    stratified_similarity_split,
)

# Split-quality reporting (how adversarial/overlapping/balanced is a split?).
from splytters.report import compare_splitters, split_report

# scikit-learn compatibility layer (CV protocol + train_test_split convenience).
from splytters.sklearn_api import (
    SplytterSplit,
    adversarial_train_test_split,
    balanced_train_test_split,
    overlap_train_test_split,
    splytter_train_test_split,
)

# Sorter introspection (lazily imported; listing pulls in no optional deps).
from splytters.sorters import list_sorters

# Stratified application of any embedding splitter / sorter (per-class).
from splytters.stratify import per_class_sort, per_class_split
from splytters.utils import (
    cluster_embeddings,
    compute_centroid,
    compute_pairwise_distances,
    compute_split_centroids,
    compute_split_similarity,
    optimized_split,
    random_split,
    validate_split_inputs,
)

__all__ = [
    # Adversarial (minimize similarity)
    "cluster_split",
    "centroid_adversarial_split",
    "distance_adversarial_split",
    "density_adversarial_split",
    "outlier_adversarial_split",
    "min_cut_split",
    "normalized_cut_split",
    "wasserstein_adversarial_split",
    "mmd_maximized_split",
    "minority_split",
    "minority_grow_split",
    "minority_route",
    "class_boundary_split",
    "decision_boundary_split",
    "maximin_split",
    "get_cluster_info",
    # Clustering-based challenging cross-validation (returns per-sample fold ids)
    "cluster_kfold",
    # Overlap (maximize similarity)
    "cluster_leak_split",
    "neighbor_coverage_split",
    "centroid_matched_split",
    "stratified_similarity_split",
    "nearest_neighbor_split",
    "duplicate_spread_split",
    "max_coverage_split",
    # Balanced (match distributions)
    "distribution_matched_split",
    "moment_matched_split",
    "histogram_matched_split",
    "stratified_random_split",
    "density_balanced_split",
    "mmd_minimized_split",
    # Baseline
    "random_split",
    # Grouped (keep related samples / near-duplicates on one side)
    "group_split",
    "deduplicated_split",
    # Curriculum (ordering-driven; pair with a splytters.sorters ranking)
    "sorted_stratified_split",
    # Stratify any embedding splitter / sorter by class (coverage-safe wrappers)
    "per_class_split",
    "per_class_sort",
    # Utilities
    "compute_pairwise_distances",
    "compute_centroid",
    "compute_split_centroids",
    "compute_split_similarity",
    "cluster_embeddings",
    "validate_split_inputs",
    "optimized_split",
    # scikit-learn compatibility
    "SplytterSplit",
    "splytter_train_test_split",
    "adversarial_train_test_split",
    "overlap_train_test_split",
    "balanced_train_test_split",
    # Framework interop
    "split_dataframe",
    "to_torch_subsets",
    "split_dataset",
    # Split-quality reporting
    "split_report",
    "compare_splitters",
    # Introspection
    "list_splitters",
    "list_sorters",
    "list_embedders",
    # Types
    "Splitter",
]


# Splitter families — the grouping returned by ``list_splitters(by_family=True)``.
# Excludes helpers (e.g. ``get_cluster_info``), the sklearn wrappers, interop
# adapters, reporting, and utilities; see ``__all__`` for the full surface.
_SPLITTER_FAMILIES: dict[str, list[str]] = {
    "adversarial": [
        "cluster_split",
        "centroid_adversarial_split",
        "distance_adversarial_split",
        "density_adversarial_split",
        "outlier_adversarial_split",
        "min_cut_split",
        "normalized_cut_split",
        "wasserstein_adversarial_split",
        "mmd_maximized_split",
        "minority_split",
        "minority_grow_split",
        "class_boundary_split",
        "decision_boundary_split",
        "maximin_split",
    ],
    "overlap": [
        "cluster_leak_split",
        "neighbor_coverage_split",
        "centroid_matched_split",
        "stratified_similarity_split",
        "nearest_neighbor_split",
        "duplicate_spread_split",
        "max_coverage_split",
    ],
    "balanced": [
        "distribution_matched_split",
        "moment_matched_split",
        "histogram_matched_split",
        "stratified_random_split",
        "density_balanced_split",
        "mmd_minimized_split",
    ],
    "baseline": [
        "random_split",
    ],
    "grouped": [
        "group_split",
        "deduplicated_split",
    ],
}


def list_splitters(by_family: bool = False) -> list[str] | dict[str, list[str]]:
    """Return the names of all available splitter functions.

    Args:
        by_family: if True, return a dict mapping each family
            ("adversarial", "overlap", "balanced", "baseline", "grouped") to its
            list of splitter names. If False (default), return a flat list of
            every splitter name, in family order.

    Returns:
        A flat list of names, or a dict of family -> names.
    """
    if by_family:
        return {family: list(names) for family, names in _SPLITTER_FAMILIES.items()}
    return [name for names in _SPLITTER_FAMILIES.values() for name in names]
