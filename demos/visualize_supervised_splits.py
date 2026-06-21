"""
Visualize supervised (label-aware) train/test splits on labeled 2D data.

Companion to visualize_splits.py, which only covers unsupervised splitters.
The splitters here need class labels ``y``, so this script uses the labeled
distribution generators (each returns ``(X, y)``).

Encoding in every cell:
    * colour  = split  -> train is blue, test is orange (same as the other demo)
    * marker  = class  -> circle / triangle / square / ... per label

Usage (run from the repo root):
    python demos/visualize_supervised_splits.py                 # interactive
    python demos/visualize_supervised_splits.py --save out.png  # save to file
    python demos/visualize_supervised_splits.py --distribution moons
    python demos/visualize_supervised_splits.py --splitter class_bndry_cen
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import splytters

# The labeled 2D generators live in ../scripts; make them importable
# regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from generate_test_2d import LABELED_GENERATORS  # noqa: E402

MARKERS = ["o", "^", "s", "D", "P", "X", "v", "*"]


def _centroid_order(X):
    """An ordering of samples by distance from the global centroid (near first).

    Gives sorted_stratified_split a concrete, geometry-driven curriculum order
    so it can be visualized like the other supervised splitters.
    """
    distances = np.linalg.norm(X - X.mean(axis=0), axis=1)
    return np.argsort(distances)


# Supervised splitters. Each entry is (name, fn) where fn(X, y) -> (train, test).
SUPERVISED_SPLITTERS = [
    ("minority", lambda X, y: splytters.minority_split(X, y, n_clusters=8)),
    ("class_bndry_cen",
     lambda X, y: splytters.class_boundary_split(X, y, reference="centroids")),
    ("class_bndry_smp",
     lambda X, y: splytters.class_boundary_split(X, y, reference="samples")),
    ("strat_random", lambda X, y: splytters.stratified_random_split(X, y)),
    ("cluster_subset_sum",
     lambda X, y: splytters.cluster_split(X, strategy="subset_sum", y=y, n_clusters=8)),
    ("sorted_strat",
     lambda X, y: splytters.sorted_stratified_split(_centroid_order(X), y)),
]


def get_splitters(name=None):
    if name:
        return [(n, fn) for n, fn in SUPERVISED_SPLITTERS if n == name]
    return SUPERVISED_SPLITTERS


def get_distributions(name=None):
    if name:
        return {name: LABELED_GENERATORS[name]}
    return LABELED_GENERATORS


def run_split(splitter_fn, X, y):
    try:
        train_idx, test_idx = splitter_fn(X, y)
        return np.asarray(train_idx), np.asarray(test_idx)
    except Exception as e:
        print(f"  split failed: {e}")
        return None, None


def _plot_cell(ax, X, y, train_idx, test_idx):
    classes = np.unique(y)
    in_train = np.zeros(len(X), dtype=bool)
    in_train[train_idx] = True
    for cls_i, cls in enumerate(classes):
        marker = MARKERS[cls_i % len(MARKERS)]
        cls_mask = y == cls
        for split_mask, colour in ((in_train, "tab:blue"), (~in_train, "tab:orange")):
            sel = cls_mask & split_mask
            if sel.any():
                ax.scatter(X[sel, 0], X[sel, 1], s=6, alpha=0.6,
                           c=colour, marker=marker, linewidths=0)


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
        X, y = gen_fn()
        for col, (split_name, split_fn) in enumerate(split_methods):
            current += 1
            print(f"[{current}/{total}] {dist_name} x {split_name}")
            ax = axes[row, col]
            train_idx, test_idx = run_split(split_fn, X, y)

            if train_idx is not None:
                _plot_cell(ax, X, y, train_idx, test_idx)
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

    # Figure-level legend: colour = split, marker = class.
    legend_handles = [
        Line2D([], [], marker="o", linestyle="", color="tab:blue", label="train"),
        Line2D([], [], marker="o", linestyle="", color="tab:orange", label="test"),
        Line2D([], [], marker="o", linestyle="", color="gray", label="class 0"),
        Line2D([], [], marker="^", linestyle="", color="gray", label="class 1"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize supervised splitter results on labeled 2D data"
    )
    parser.add_argument("--save", type=str, default=None, help="Save figure to path")
    parser.add_argument(
        "--splitter",
        choices=[name for name, _ in SUPERVISED_SPLITTERS],
        default=None,
        help="Only show this splitter",
    )
    parser.add_argument(
        "--distribution",
        choices=list(LABELED_GENERATORS.keys()),
        default=None,
        help="Only show this distribution",
    )
    args = parser.parse_args()

    split_methods = get_splitters(args.splitter)
    distributions = get_distributions(args.distribution)

    print(f"Distributions: {list(distributions.keys())}")
    print(f"Splitters: {[name for name, _ in split_methods]}")
    visualize(distributions, split_methods, save_path=args.save)


if __name__ == "__main__":
    main()
