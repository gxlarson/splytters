"""
Visualize train/test splits produced by every splitter on every 2D distribution.

Generates a grid: rows = distributions, columns = splitters.
Train points are blue, test points are orange.

Usage:
    python visualize_splits.py                  # show interactive plot
    python visualize_splits.py --save out.png   # save to file
    python visualize_splits.py --category adversarial   # only adversarial splitters
    python visualize_splits.py --distribution moons     # only moons data
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from generate_test_2d import ALL_GENERATORS
import splitters

# Splitters grouped by category. Each entry is (name, callable).
# stratified_random_split is excluded because it requires labels.
SPLITTER_GROUPS = {
    "baseline": [
        ("random", splitters.random_split),
    ],
    "adversarial": [
        ("cluster", splitters.cluster_split),
        ("centroid_adv", splitters.centroid_adversarial_split),
        ("distance_adv", splitters.distance_adversarial_split),
        ("density_adv", splitters.density_adversarial_split),
        ("outlier_adv", splitters.outlier_adversarial_split),
        ("min_cut", splitters.min_cut_split),
        ("normalized_cut", splitters.normalized_cut_split),
    ],
    "overlap": [
        ("cluster_leak", splitters.cluster_leak_split),
        ("neighbor_cov", splitters.neighbor_coverage_split),
        ("centroid_match", splitters.centroid_matched_split),
        ("strat_sim", splitters.stratified_similarity_split),
        ("nearest_nbr", splitters.nearest_neighbor_split),
        ("dup_spread", splitters.duplicate_spread_split),
        ("max_coverage", splitters.max_coverage_split),
    ],
    "balanced": [
        ("dist_matched", splitters.distribution_matched_split),
        ("moment_matched", splitters.moment_matched_split),
        ("hist_matched", splitters.histogram_matched_split),
        ("density_bal", splitters.density_balanced_split),
        ("mmd_min", splitters.mmd_minimized_split),
    ],
}


def get_splitters(category=None):
    baseline = SPLITTER_GROUPS["baseline"]
    if category:
        if category == "baseline":
            return baseline
        return baseline + SPLITTER_GROUPS[category]
    return [(name, fn) for group in SPLITTER_GROUPS.values() for name, fn in group]


def get_distributions(name=None):
    if name:
        return {name: ALL_GENERATORS[name]}
    return ALL_GENERATORS


def run_split(splitter_fn, data):
    try:
        train_idx, test_idx = splitter_fn(data)
        return train_idx, test_idx
    except Exception as e:
        print(f"  split failed: {e}")
        return None, None


def visualize(distributions, split_methods, save_path=None):
    n_rows = len(distributions)
    n_cols = len(split_methods)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )

    dist_items = list(distributions.items())

    total = n_rows * n_cols
    current = 0
    for row, (dist_name, gen_fn) in enumerate(dist_items):
        data = gen_fn()
        for col, (split_name, split_fn) in enumerate(split_methods):
            current += 1
            print(f"[{current}/{total}] {dist_name} x {split_name}")
            ax = axes[row, col]
            train_idx, test_idx = run_split(split_fn, data)

            if train_idx is not None:
                train_idx = np.asarray(train_idx)
                test_idx = np.asarray(test_idx)
                ax.scatter(data[train_idx, 0], data[train_idx, 1],
                           s=4, alpha=0.5, c="tab:blue", label="train")
                ax.scatter(data[test_idx, 0], data[test_idx, 1],
                           s=4, alpha=0.5, c="tab:orange", label="test")
            else:
                ax.text(0.5, 0.5, "failed", transform=ax.transAxes,
                        ha="center", va="center", color="red")

            if row == 0:
                ax.set_title(split_name, fontsize=8)
            if col == 0:
                ax.set_ylabel(dist_name, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

    # single legend for the whole figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize splitter results on 2D data")
    parser.add_argument("--save", type=str, default=None, help="Save figure to path")
    parser.add_argument(
        "--category",
        choices=list(SPLITTER_GROUPS.keys()),
        default=None,
        help="Only show splitters from this category",
    )
    parser.add_argument(
        "--distribution",
        choices=list(ALL_GENERATORS.keys()),
        default=None,
        help="Only show this distribution",
    )
    args = parser.parse_args()

    split_methods = get_splitters(args.category)
    distributions = get_distributions(args.distribution)

    print(f"Distributions: {list(distributions.keys())}")
    print(f"Splitters: {[name for name, _ in split_methods]}")
    visualize(distributions, split_methods, save_path=args.save)


if __name__ == "__main__":
    main()
