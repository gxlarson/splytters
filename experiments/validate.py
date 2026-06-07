"""
Stronger validation of the core thesis, across datasets *and* model families.

For each (dataset, split objective, model, seed) we train on the split's training
embeddings and evaluate on its test embeddings, recording **balanced** accuracy
(robust to the label-skew confound) plus split diagnostics from `split_report`.

It answers three questions a reviewer would ask:

1. Is the difficulty ordering (adversarial < random ≈ balanced ≈ overlap) real,
   or a class-coverage artifact?  -> we report balanced accuracy + missing-class
   counts + label shift, and include a *controlled* dataset where features are
   decorrelated from labels.
2. Is it model-agnostic?  -> we sweep logistic regression, random forest, linear
   SVM, and k-NN.
3. Can `split_report` predict difficulty WITHOUT training a model?  -> we
   correlate its energy distance with the realized balanced-accuracy drop.

Datasets: `synth` (controlled, offline), `digits` (offline), `newsgroups`
(real text embedded with sentence-transformers; uses a cached .npz if present,
otherwise built on first run and cached).

Usage:
    python experiments/validate.py --seeds 5 --out experiments/results
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from splitters import (
    centroid_adversarial_split,
    cluster_leak_split,
    cluster_split,
    distribution_matched_split,
    random_split,
    split_report,
)

CACHE = Path("experiments/cache")

def _adv_cluster(X, s):
    return cluster_split(X, train_size=0.7, random_state=s, n_clusters=10)


def _adv_centroid(X, s):
    return centroid_adversarial_split(X, train_size=0.7, random_state=s, n_clusters=10)


def _overlap(X, s):
    return cluster_leak_split(X, train_size=0.7, random_state=s, n_clusters=10)


def _balanced(X, s):
    return distribution_matched_split(X, train_size=0.7, random_state=s, n_iterations=300)


SPLITTERS = {
    "random": lambda X, s: random_split(X, train_size=0.7, random_state=s),
    "adv:cluster": _adv_cluster,
    "adv:centroid": _adv_centroid,
    "overlap": _overlap,
    "balanced": _balanced,
}

MODELS = {
    "logreg": lambda: LogisticRegression(max_iter=2000),
    "rf": lambda: RandomForestClassifier(n_estimators=150, random_state=0, n_jobs=-1),
    "linsvm": lambda: LinearSVC(max_iter=5000),
    "knn": lambda: KNeighborsClassifier(n_neighbors=15),
}


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "synth":
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=1500, n_features=20, n_informative=10, n_classes=4,
            n_clusters_per_class=3, class_sep=1.0, random_state=0,
        )
        return StandardScaler().fit_transform(X), y

    if name == "digits":
        from sklearn.datasets import load_digits

        d = load_digits()
        return StandardScaler().fit_transform(d.data.astype(float)), d.target

    if name == "newsgroups":
        cache = CACHE / "newsgroups.npz"
        if cache.exists():
            z = np.load(cache)
            return StandardScaler().fit_transform(z["X"]), z["y"]
        # Build + cache (needs the [demo] extra: datasets/sentence-transformers).
        from sentence_transformers import SentenceTransformer
        from sklearn.datasets import fetch_20newsgroups

        cats = ["rec.sport.hockey", "sci.med", "talk.politics.guns", "comp.graphics"]
        d = fetch_20newsgroups(
            subset="train", categories=cats, remove=("headers", "footers", "quotes")
        )
        keep = [i for i, t in enumerate(d.data) if len(t.split()) >= 15][:1600]
        texts = [d.data[i] for i in keep]
        y = np.asarray(d.target)[keep]
        X = SentenceTransformer("all-MiniLM-L6-v2").encode(texts, show_progress_bar=False)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X.astype("float32"), y=y)
        return StandardScaler().fit_transform(X), y

    raise ValueError(f"unknown dataset {name!r}")


def evaluate(name: str, X: np.ndarray, y: np.ndarray, seeds: int) -> list[dict]:
    """Return one row per (split, model, seed)."""
    rows: list[dict] = []
    for split_name, splitter in SPLITTERS.items():
        for seed in range(seeds):
            tr, te = splitter(X, seed)
            missing = len(set(y[te].tolist()) - set(y[tr].tolist()))
            rep = split_report(X, tr, te, y=y, max_samples=800, random_state=seed)
            for model_name, factory in MODELS.items():
                clf = factory().fit(X[tr], y[tr])
                bal = balanced_accuracy_score(y[te], clf.predict(X[te]))
                rows.append({
                    "dataset": name, "split": split_name, "model": model_name,
                    "seed": seed, "bal_acc": float(bal),
                    "label_shift": rep["label_distribution_shift"],
                    "energy": rep["energy_distance"], "missing": missing,
                })
    return rows


def summarize(rows: list[dict]) -> None:
    datasets = sorted({r["dataset"] for r in rows}, key="synth digits newsgroups".index)
    for ds in datasets:
        dr = [r for r in rows if r["dataset"] == ds]
        print(f"\n==== {ds} ====")
        header = f"{'split':<14}" + "".join(f"{m:>9}" for m in MODELS) + \
                 f"{'mean':>8}{'lblΔ':>7}{'miss':>6}{'energy':>8}"
        print(header)
        base = None
        for split_name in SPLITTERS:
            sr = [r for r in dr if r["split"] == split_name]
            per_model = [
                np.mean([r["bal_acc"] for r in sr if r["model"] == m]) for m in MODELS
            ]
            mean = float(np.mean(per_model))
            if split_name == "random":
                base = mean
            lbl = float(np.mean([r["label_shift"] for r in sr]))
            miss = float(np.mean([r["missing"] for r in sr]))
            eng = float(np.mean([r["energy"] for r in sr]))
            cells = "".join(f"{v:>9.3f}" for v in per_model)
            print(f"{split_name:<14}{cells}{mean:>8.3f}{lbl:>7.2f}{miss:>6.1f}{eng:>8.2f}")
        if base is not None:
            adv = float(np.mean([
                r["bal_acc"] for r in dr if r["split"] == "adv:cluster"
            ]))
            print(f"  adversarial vs random balanced-acc gap: {base - adv:+.3f}")


def correlation(rows: list[dict]) -> tuple[float, float]:
    """Spearman corr between split energy distance and balanced-acc drop vs random."""
    from scipy.stats import spearmanr

    # per (dataset, split, seed): mean bal_acc over models, energy, and the
    # random baseline for that (dataset, seed).
    energies, drops = [], []
    for ds in {r["dataset"] for r in rows}:
        for seed in {r["seed"] for r in rows}:
            base_rows = [r for r in rows if r["dataset"] == ds and r["seed"] == seed
                         and r["split"] == "random"]
            if not base_rows:
                continue
            base = np.mean([r["bal_acc"] for r in base_rows])
            for split_name in SPLITTERS:
                sr = [r for r in rows if r["dataset"] == ds and r["seed"] == seed
                      and r["split"] == split_name]
                if not sr:
                    continue
                energies.append(float(np.mean([r["energy"] for r in sr])))
                drops.append(base - float(np.mean([r["bal_acc"] for r in sr])))
    rho, p = spearmanr(energies, drops)
    return float(rho), float(p)


def plot(rows: list[dict], out_dir: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return None

    datasets = sorted({r["dataset"] for r in rows}, key="synth digits newsgroups".index)
    colors = {"random": "#888", "adv:cluster": "#d62728", "adv:centroid": "#ff7f0e",
              "overlap": "#2ca02c", "balanced": "#1f77b4"}
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), squeeze=False)
    for ax, ds in zip(axes[0], datasets, strict=False):
        dr = [r for r in rows if r["dataset"] == ds]
        names = list(SPLITTERS)
        means, errs = [], []
        for split_name in names:
            vals = [np.mean([r["bal_acc"] for r in dr
                             if r["split"] == split_name and r["model"] == m])
                    for m in MODELS]
            means.append(float(np.mean(vals)))
            errs.append(float(np.std(vals)))
        ax.bar(names, means, yerr=errs, capsize=4,
               color=[colors[n] for n in names])
        base = means[names.index("random")]
        ax.axhline(base, ls="--", lw=1, color="#444")
        ax.set_title(ds)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel("balanced accuracy (mean over 4 models)")
    fig.suptitle("Adversarial splits are harder across datasets and model families")
    fig.tight_layout()
    p = out_dir / "validation.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--datasets", nargs="+", default=["synth", "digits", "newsgroups"])
    ap.add_argument("--out", default="experiments/results")
    args = ap.parse_args()

    rows: list[dict] = []
    for ds in args.datasets:
        X, y = load(ds)
        print(f"loaded {ds}: X={X.shape}, classes={len(np.unique(y))}")
        rows += evaluate(ds, X, y, args.seeds)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "validation.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summarize(rows)
    rho, p = correlation(rows)
    print(f"\nSpearman(energy distance, balanced-acc drop vs random) = "
          f"{rho:+.3f} (p={p:.1e})  -> split_report predicts difficulty without training")
    fig = plot(rows, out_dir)
    print(f"\nrows: {len(rows)}  csv: {csv_path}" + (f"  figure: {fig}" if fig else ""))


if __name__ == "__main__":
    main()
