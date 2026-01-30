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
    cluster_split,
    centroid_adversarial_split,
    distance_adversarial_split,
    density_adversarial_split,
    outlier_adversarial_split,
    min_cut_split,
    normalized_cut_split,
    get_cluster_info,
)

from splitters.overlap import (
    cluster_leak_split,
    neighbor_coverage_split,
    centroid_matched_split,
    stratified_similarity_split,
    nearest_neighbor_split,
    duplicate_spread_split,
    max_coverage_split,
)

from splitters.balanced import (
    distribution_matched_split,
    moment_matched_split,
    histogram_matched_split,
    stratified_random_split,
    density_balanced_split,
    mmd_minimized_split,
)

from splitters.utils import (
    random_split,
    compute_pairwise_distances,
    compute_centroid,
    compute_split_centroids,
    compute_split_similarity,
    cluster_embeddings,
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
]
