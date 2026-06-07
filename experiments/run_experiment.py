"""
Illustrative experiment: does the *split objective* change evaluation difficulty?

For a dataset of embeddings + labels, we build train/test splits with each
objective (random / adversarial / overlap / balanced), train a cheap downstream
classifier (logistic regression on the embeddings) on each, and measure test
accuracy across several seeds. The expected, motivating result:

    adversarial  <  balanced ≈ random  <  overlap

i.e. the *same model and data* look much weaker under an adversarial split and
much stronger under an overlap split — so a single random number can badly
misestimate generalization.

The default dataset (`digits`) is fully offline (ships with scikit-learn), so
this reproduces with no downloads. Pass `--dataset trec` (needs the `[demo]`
extra) to embed real text with sentence-transformers.

Usage
-----
    python experiments/run_experiment.py --dataset digits --seeds 10 \
        --out experiments/results
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from splitters import (
    cluster_leak_split,
    cluster_split,
    distribution_matched_split,
    random_split,
    split_report,
)

# Splitter per objective. All respond to ``random_state`` so seeds vary them.
SPLITTERS = {
    "random": random_split,
    "adversarial": cluster_split,
    "overlap": cluster_leak_split,
    "balanced": distribution_matched_split,
}


def load_embeddings(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (embeddings, labels) for a named dataset."""
    if name == "digits":
        from sklearn.datasets import load_digits

        d = load_digits()
        X = StandardScaler().fit_transform(d.data.astype(float))
        return X, d.target

    if name == "synth":
        # Controlled covariate-shift benchmark: feature clusters are deliberately
        # NOT aligned to labels (n_clusters_per_class > 1), so a feature-based
        # adversarial split induces genuine covariate shift WITHOUT dropping or
        # skewing classes. This isolates "harder because dissimilar" from the
        # "harder because a class is missing/under-represented" confound that
        # appears on label-clustered data like `digits`.
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=1500, n_features=20, n_informative=10, n_classes=4,
            n_clusters_per_class=3, class_sep=1.0, random_state=0,
        )
        return StandardScaler().fit_transform(X), y

    if name == "trec":  # pragma: no cover - needs network + [demo] extra
        from datasets import load_dataset
        from sentence_transformers import SentenceTransformer

        ds = load_dataset("trec", split="train[:2000]")
        texts = ds["text"]
        y = np.asarray(ds["coarse_label"])
        X = SentenceTransformer("all-MiniLM-L6-v2").encode(texts)
        return StandardScaler().fit_transform(X), y

    raise ValueError(f"unknown dataset {name!r} (try 'digits', 'synth', or 'trec')")


def run(
    X: np.ndarray, y: np.ndarray, seeds: int, train_size: float
) -> dict[str, dict[str, float]]:
    """Train/evaluate every splitter across seeds; return summary stats.

    Records *balanced* accuracy and label-coverage diagnostics alongside raw
    accuracy, so a difficulty gap caused by genuine covariate shift can be told
    apart from one caused by a class being dropped or skewed out of training.
    """
    cols = ("acc", "bal_acc", "mmd", "energy", "label_shift", "missing_train")
    bucket: dict[str, dict[str, list[float]]] = {
        k: {c: [] for c in cols} for k in SPLITTERS
    }

    for name, splitter in SPLITTERS.items():
        for seed in range(seeds):
            train_idx, test_idx = splitter(X, train_size=train_size, random_state=seed)
            clf = LogisticRegression(max_iter=2000)
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[test_idx])
            bucket[name]["acc"].append(float(accuracy_score(y[test_idx], pred)))
            bucket[name]["bal_acc"].append(
                float(balanced_accuracy_score(y[test_idx], pred))
            )
            # classes present in test but absent from train (pure coverage loss)
            missing = len(set(y[test_idx].tolist()) - set(y[train_idx].tolist()))
            bucket[name]["missing_train"].append(float(missing))
            rep = split_report(
                X, train_idx, test_idx, y=y, max_samples=1000, random_state=seed
            )
            bucket[name]["mmd"].append(rep["mmd_rbf"])
            bucket[name]["energy"].append(rep["energy_distance"])
            bucket[name]["label_shift"].append(rep["label_distribution_shift"])

    summary: dict[str, dict[str, float]] = {}
    for name in SPLITTERS:
        b = bucket[name]
        summary[name] = {
            "mean_acc": statistics.fmean(b["acc"]),
            "std_acc": statistics.pstdev(b["acc"]) if len(b["acc"]) > 1 else 0.0,
            "mean_bal_acc": statistics.fmean(b["bal_acc"]),
            "std_bal_acc": statistics.pstdev(b["bal_acc"]) if len(b["bal_acc"]) > 1 else 0.0,
            "mean_mmd": statistics.fmean(b["mmd"]),
            "mean_energy": statistics.fmean(b["energy"]),
            "mean_label_shift": statistics.fmean(b["label_shift"]),
            "mean_missing_train": statistics.fmean(b["missing_train"]),
            "n_seeds": len(b["acc"]),
        }
    return summary


def save_results(summary: dict[str, dict[str, float]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "accuracy_by_split.csv"
    fields = [
        "mean_acc", "std_acc", "mean_bal_acc", "std_bal_acc",
        "mean_mmd", "mean_energy", "mean_label_shift", "mean_missing_train",
        "n_seeds",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["splitter", *fields])
        for name, s in summary.items():
            w.writerow([name, *(f"{s[k]:.6g}" for k in fields)])
    return csv_path


def plot_results(summary: dict[str, dict[str, float]], out_dir: Path, dataset: str) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return None

    names = list(summary)
    raw = [summary[n]["mean_acc"] for n in names]
    raw_err = [summary[n]["std_acc"] for n in names]
    bal = [summary[n]["mean_bal_acc"] for n in names]
    bal_err = [summary[n]["std_bal_acc"] for n in names]

    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, raw, w, yerr=raw_err, capsize=4, color="#d62728",
           label="raw accuracy")
    ax.bar(x + w / 2, bal, w, yerr=bal_err, capsize=4, color="#1f77b4",
           label="balanced accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"Difficulty by split objective ({dataset})")
    # Annotate label-distribution shift so coverage effects are visible.
    for xi, n in zip(x, names, strict=False):
        ax.text(xi, 0.03, f"lblΔ={summary[n]['mean_label_shift']:.2f}",
                ha="center", fontsize=7, color="#333")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig_path = out_dir / "accuracy_by_split.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="digits")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--train-size", type=float, default=0.7)
    ap.add_argument("--out", default="experiments/results")
    args = ap.parse_args()

    X, y = load_embeddings(args.dataset)
    print(f"Loaded {args.dataset}: X={X.shape}, classes={len(np.unique(y))}")
    summary = run(X, y, seeds=args.seeds, train_size=args.train_size)

    out_dir = Path(args.out)
    csv_path = save_results(summary, out_dir)
    fig_path = plot_results(summary, out_dir, args.dataset)

    hdr = f"{'splitter':<12}{'acc':>7}{'bal_acc':>9}{'lbl_shift':>10}{'miss_tr':>8}{'energy':>8}"
    print("\n" + hdr)
    for name, s in summary.items():
        print(f"{name:<12}{s['mean_acc']:>7.3f}{s['mean_bal_acc']:>9.3f}"
              f"{s['mean_label_shift']:>10.3f}{s['mean_missing_train']:>8.1f}"
              f"{s['mean_energy']:>8.3f}")

    r, a = summary["random"], summary["adversarial"]
    print(
        f"\nadversarial vs random:"
        f"  raw acc gap {r['mean_acc'] - a['mean_acc']:+.3f}"
        f"  |  balanced acc gap {r['mean_bal_acc'] - a['mean_bal_acc']:+.3f}"
        f"  |  energy x{(a['mean_energy'] / max(r['mean_energy'], 1e-9)):.1f}"
    )
    if a["mean_missing_train"] < 0.5 and a["mean_label_shift"] < 0.30:
        print("  -> classes are preserved & label shift is small: the gap reflects "
              "genuine COVARIATE shift (clean validation of the core idea).")
    else:
        print("  -> NOTE: label shift / missing classes are non-trivial here, so the "
              "raw-acc gap conflates covariate shift with label coverage; read the "
              "balanced-acc gap (and try --dataset synth for a clean test).")
    print(f"results: {csv_path}")
    if fig_path:
        print(f"figure:  {fig_path}")


if __name__ == "__main__":
    main()
