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

from splitters.adversarial import (
    centroid_adversarial_split,
    cluster_split,
    density_adversarial_split,
    distance_adversarial_split,
    get_cluster_info,
    min_cut_split,
    normalized_cut_split,
    outlier_adversarial_split,
    wasserstein_adversarial_split,
)
from splitters.balanced import (
    density_balanced_split,
    distribution_matched_split,
    histogram_matched_split,
    mmd_minimized_split,
    moment_matched_split,
    stratified_random_split,
)

# Framework interop (pandas / torch / HuggingFace datasets). Heavy deps are
# imported lazily inside each helper, so this import stays dependency-light.
from splitters.interop import (
    split_dataframe,
    split_dataset,
    to_torch_subsets,
)
from splitters.overlap import (
    centroid_matched_split,
    cluster_leak_split,
    duplicate_spread_split,
    max_coverage_split,
    nearest_neighbor_split,
    neighbor_coverage_split,
    stratified_similarity_split,
)

# Split-quality reporting (how adversarial/overlapping/balanced is a split?).
from splitters.report import compare_splitters, split_report

# scikit-learn compatibility layer (CV protocol + train_test_split convenience).
from splitters.sklearn_api import (
    SplytterSplit,
    adversarial_train_test_split,
    balanced_train_test_split,
    overlap_train_test_split,
    splytter_train_test_split,
)
from splitters.utils import (
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
    "get_cluster_info",
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
]
