"""Compare a faithful DEEP CLUSTER minority split against the in-library backends.

Runs, on TREC (6-class), the same head-to-head as demos/minority_clustering_comparison.py
but adds the real thing: ``deepcluster`` (torch/transformers). Every method's cluster
labels are routed through the SAME ``splytters.minority_route`` so we isolate the effect
of the *clustering*, and every split is scored with the SAME logistic-regression probe on
the SAME frozen embeddings so we isolate the effect of the *split*.

Question: does faithful DEEP CLUSTER produce more label-diverse clusters / a harder
minority split than our cheap deepcluster-lite surrogate?

Run (smoke):  python run_trec.py --smoke
Run (real, in screen):  python run_trec.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from deepcluster import deepcluster_labels
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from splytters import minority_route
from splytters.adversarial import _minority_cluster_labels

HERE = Path(__file__).parent
ENCODER = "sentence-transformers/all-distilroberta-v1"
N_CLUSTERS = 10


def load_trec():
    from datasets import load_dataset

    # Recent `datasets` dropped legacy scripts; use the auto-converted parquet branch.
    ds = load_dataset("CogComp/trec", revision="refs/convert/parquet")
    col = "coarse_label" if "coarse_label" in ds["train"].features else "label-coarse"
    texts, y = [], []
    for split in ("train", "test"):
        texts += list(ds[split]["text"])
        y += list(ds[split][col])
    return texts, np.asarray(y)


def frozen_embeddings(texts: list[str]) -> np.ndarray:
    import hashlib

    from sentence_transformers import SentenceTransformer

    # Key the cache by encoder + text content so the smoke subset and the full
    # set don't collide, and so switching ENCODER can't silently serve stale
    # embeddings from a different model.
    key = hashlib.sha1((ENCODER + "\n" + "\n".join(texts)).encode()).hexdigest()[:12]
    cache = HERE / "results" / f"trec_frozen_emb_{key}.npy"
    if cache.exists():
        return np.load(cache)
    emb = SentenceTransformer(ENCODER).encode(texts, batch_size=64,
                                              show_progress_bar=False)
    np.save(cache, emb)
    return emb


def evaluate(labels: np.ndarray, y: np.ndarray, Xeval: np.ndarray, seed: int = 0):
    """Route a clustering to a split, then score its difficulty on frozen features."""
    pure, ent_num, uniq = 0, 0.0, np.unique(labels)
    for c in uniq:
        _, counts = np.unique(y[labels == c], return_counts=True)
        if len(counts) == 1:
            pure += 1
        p = counts / counts.sum()
        ent_num += counts.sum() * -(p * np.log(p)).sum()
    tr, te = minority_route(labels, y)
    n_te = len(te)
    clf = LogisticRegression(max_iter=2000).fit(Xeval[tr], y[tr])
    seen = np.isin(y[te], np.unique(y[tr]))
    test_acc = clf.score(Xeval[te][seen], y[te][seen]) if seen.any() else np.nan
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y))
    r_te, r_tr = perm[:n_te], perm[n_te:]
    rclf = LogisticRegression(max_iter=2000).fit(Xeval[r_tr], y[r_tr])
    rseen = np.isin(y[r_te], np.unique(y[r_tr]))
    rand_acc = rclf.score(Xeval[r_te][rseen], y[r_te][rseen]) if rseen.any() else np.nan
    return {
        "test_frac": n_te / len(y),
        "pure_frac": pure / len(uniq),
        "label_entropy": ent_num / len(y),
        "test_acc": test_acc,
        "acc_drop": test_acc - rand_acc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny subset, fast wiring check")
    ap.add_argument("--epochs", type=int, default=1,
                    help="epochs for the pseudo-label deep-cluster pass")
    ap.add_argument("--task-finetune", action="store_true",
                    help="fully faithful: task fine-tune for clustering #1 too")
    ap.add_argument("--task-epochs", type=int, default=3,
                    help="epochs for the clustering-#1 task fine-tune (faithful only)")
    args = ap.parse_args()

    texts, y = load_trec()
    if args.smoke:
        rng = np.random.RandomState(0)
        keep = np.sort(rng.choice(len(y), size=300, replace=False))
        texts, y = [texts[i] for i in keep], y[keep]
    print(f"TREC: n={len(y)}, {len(np.unique(y))} classes")

    Xeval = normalize(frozen_embeddings(texts)).astype(np.float32)

    rows = {}
    # In-library backends (cheap): cluster via the library, route via minority_route.
    for method in ("kmeans", "ward", "deepcluster-lite"):
        labels = _minority_cluster_labels(Xeval, method, N_CLUSTERS, 0)
        rows[method] = evaluate(labels, y, Xeval)
        print(f"[{method}] done")

    # Faithful DEEP CLUSTER (heavy): its own clustering, same routing + eval.
    reps: dict = {}
    dc_labels = deepcluster_labels(
        texts, Xeval, y=y, n_clusters=N_CLUSTERS, n_iters=1,
        task_finetune=args.task_finetune, epochs=args.epochs,
        task_epochs=args.task_epochs, max_len=64, reps_out=reps,
    )
    tag = "deepcluster-faithful" if args.task_finetune else "deepcluster-semifaithful"
    rows[tag] = evaluate(dc_labels, y, Xeval)
    print(f"[{tag}] done")

    # Decisive check: cluster the task-fine-tuned [CLS] directly. Plain clustering
    # should go label-homogeneous here (what DEEP CLUSTER exists to undo), while the
    # deepcluster-faithful row above re-clusters after the pseudo-label pass.
    if "task_rep" in reps:
        Xft = normalize(reps["task_rep"]).astype(np.float32)
        for m in ("kmeans", "ward"):
            labels = _minority_cluster_labels(Xft, m, N_CLUSTERS, 0)
            rows[f"{m}@ft_cls"] = evaluate(labels, y, Xeval)
            print(f"[{m}@ft_cls] done")

    hdr = f"{'method':<26}{'test_frac':>10}{'pure_frac':>10}{'entropy':>9}" \
          f"{'test_acc':>10}{'acc_drop':>10}"
    lines = [hdr, "-" * len(hdr)]
    for m, a in rows.items():
        lines.append(f"{m:<26}{a['test_frac']:>10.3f}{a['pure_frac']:>10.3f}"
                     f"{a['label_entropy']:>9.3f}{a['test_acc']:>10.3f}"
                     f"{a['acc_drop']:>10.3f}")
    out = "\n".join(lines)
    print("\n" + out)
    dst = HERE / "results" / ("trec_smoke.txt" if args.smoke else "trec_comparison.txt")
    dst.write_text(out + "\n")
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
