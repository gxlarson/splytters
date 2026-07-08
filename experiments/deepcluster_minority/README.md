# DEEP CLUSTER minority split (Option A)

A **faithful** implementation of the clustering half of the minority-examples split
from Reif & Schwartz (2023), *"Fighting Bias with Bias"* ([ACL Findings](https://aclanthology.org/2023.findings-acl.833),
[arXiv:2305.18917](https://arxiv.org/abs/2305.18917)), using their actual method —
**DEEP CLUSTER** (Caron et al., 2018) — which trains an encoder on cluster
pseudo-labels to produce *label-diverse* clusters.

This is the heavy counterpart to the in-library `minority_split` clustering backends
(`kmeans` / `ward` / `deepcluster-lite`). It exists to answer: **does the real
DEEP CLUSTER produce more label-diverse clusters and a harder minority split than our
cheap `deepcluster-lite` surrogate?**

## Why it lives outside the package

`splytters` is a pure, gradient-free library. DEEP CLUSTER needs `torch`,
`transformers`, task data, and a training loop — so it lives here and only *consumes*
the library. It produces a cluster assignment and hands it to
`splytters.minority_route(cluster_labels, y, minority_labels=...)`, so the train/test
routing (including footnote-10) is **byte-identical** to `minority_split`.

## Method

1. **Clustering #1 source**
   - *semi-faithful* (default): frozen sentence embeddings — one fine-tune total.
   - *faithful* (`--task-finetune`): a task-fine-tuned encoder's `[CLS]` — two fine-tunes.
2. Ward-cluster → pseudo-labels.
3. **Deep-cluster iteration**: fine-tune a *fresh* `roberta-base` for one epoch to
   predict the pseudo-labels; take its `[CLS]` representation.
4. Ward-cluster that representation → final labels.
5. `minority_route` → bias-amplified train/test split.

## Run

```bash
# fast wiring check (300 examples, 1 epoch)
python run_trec.py --smoke

# real run — do this in a screen session; CPU fine-tuning is slow
python run_trec.py                 # semi-faithful
python run_trec.py --task-finetune # fully faithful (two fine-tunes)
```

Every method's split is scored with the same logistic-regression probe on the same
frozen embeddings, so the comparison isolates the effect of the clustering. Results
land in `results/`.

## Files
- `deepcluster.py` — the DEEP CLUSTER algorithm (all torch/transformers code).
- `run_trec.py` — driver + comparison harness.
- `requirements.txt` — heavy deps (already in the repo `.venv`).
