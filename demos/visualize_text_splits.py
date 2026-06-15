"""
Visualize splitter methods on text data.

Pipeline:
1. Load text queries from test_data/text/bank_balance_queries.txt
2. Embed with TextEmbedder (SentenceTransformer)
3. Run each splitter to get train/test indices
4. Project embeddings to 2D with UMAP
5. Plot a grid showing train/test splits, with sample indices annotated

Usage (run from the repo root):
    python demos/visualize_text_splits.py
    python demos/visualize_text_splits.py --save demos/text_splits.png
    python demos/visualize_text_splits.py --category adversarial
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import splytters
from splytters.embedders import TextEmbedder

# dim_reduce is a sibling demo helper; make it importable regardless of the
# current working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dim_reduce import reduce_tsne, reduce_umap  # noqa: E402

SPLITTER_GROUPS = {
    "baseline": [
        ("random", splytters.random_split),
    ],
    "adversarial": [
        ("cluster", splytters.cluster_split),
        ("centroid_adv", splytters.centroid_adversarial_split),
        ("distance_adv", splytters.distance_adversarial_split),
        ("density_adv", splytters.density_adversarial_split),
        ("outlier_adv", splytters.outlier_adversarial_split),
    ],
    "overlap": [
        ("cluster_leak", splytters.cluster_leak_split),
        ("neighbor_cov", splytters.neighbor_coverage_split),
        ("centroid_match", splytters.centroid_matched_split),
        ("strat_sim", splytters.stratified_similarity_split),
        ("nearest_nbr", splytters.nearest_neighbor_split),
    ],
    "balanced": [
        ("dist_matched", splytters.distribution_matched_split),
        ("moment_matched", splytters.moment_matched_split),
        ("hist_matched", splytters.histogram_matched_split),
        ("density_bal", splytters.density_balanced_split),
    ],
}

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "test_data", "text", "bank_balance_queries.txt"
)


def load_texts(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def get_splitters(category=None):
    baseline = SPLITTER_GROUPS["baseline"]
    if category:
        if category == "baseline":
            return baseline
        return baseline + SPLITTER_GROUPS[category]
    return [(name, fn) for group in SPLITTER_GROUPS.values() for name, fn in group]


def run_split(splitter_fn, embeddings):
    try:
        train_idx, test_idx = splitter_fn(embeddings)
        return np.asarray(train_idx), np.asarray(test_idx)
    except Exception as e:
        print(f"  split failed: {e}")
        return None, None


def visualize(coords_2d, split_methods, texts, save_path=None, reduce_method="umap"):
    n_cols = min(4, len(split_methods))
    n_rows = (len(split_methods) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 5 * n_rows),
        squeeze=False,
    )

    for i, (name, split_fn) in enumerate(split_methods):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]
        print(f"[{i + 1}/{len(split_methods)}] {name}")

        train_idx, test_idx = run_split(split_fn, embeddings_hd)

        if train_idx is not None:
            ax.scatter(coords_2d[train_idx, 0], coords_2d[train_idx, 1],
                       s=30, alpha=0.7, c="tab:blue", label="train", zorder=2)
            ax.scatter(coords_2d[test_idx, 0], coords_2d[test_idx, 1],
                       s=30, alpha=0.7, c="tab:orange", label="test", zorder=2)
            # annotate with original line indices
            for idx in range(len(coords_2d)):
                ax.annotate(str(idx), (coords_2d[idx, 0], coords_2d[idx, 1]),
                            fontsize=5, alpha=0.6, ha="center", va="bottom")
        else:
            ax.text(0.5, 0.5, "failed", transform=ax.transAxes,
                    ha="center", va="center", color="red")

        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # hide unused subplots
    for i in range(len(split_methods), n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row][col].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=9)

    plt.suptitle(f"Splitter Methods on Text Embeddings ({reduce_method.upper()} 2D)", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize splitters on text embeddings")
    parser.add_argument("--save", type=str, default=None, help="Save figure to path")
    parser.add_argument(
        "--category",
        choices=list(SPLITTER_GROUPS.keys()),
        default=None,
        help="Only show splitters from this category",
    )
    parser.add_argument(
        "--method",
        choices=["umap", "tsne"],
        default="umap",
        help="Dimensionality reduction method (default: umap)",
    )
    args = parser.parse_args()

    # 1. Load text
    texts = load_texts(DATA_PATH)
    print(f"Loaded {len(texts)} queries")

    # 2. Embed with SentenceTransformer
    print("Embedding texts...")
    embedder = TextEmbedder()
    embeddings_hd = embedder.embed(texts)
    print(f"Embeddings shape: {embeddings_hd.shape}")

    # 3. Reduce to 2D for visualization
    reduce_fn = reduce_tsne if args.method == "tsne" else reduce_umap
    print(f"Reducing to 2D with {args.method.upper()}...")
    coords_2d = reduce_fn(embeddings_hd, n_components=2, random_state=42)

    # 4. Run splitters and visualize
    split_methods = get_splitters(args.category)
    print(f"Splitters: {[name for name, _ in split_methods]}")
    visualize(coords_2d, split_methods, texts, save_path=args.save, reduce_method=args.method)
