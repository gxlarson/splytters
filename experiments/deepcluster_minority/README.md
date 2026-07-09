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

## Results (TREC, 6-class, n=5952)

`results/trec_comparison.txt`, all methods routed through the same `minority_route`
and scored with the same logistic-regression probe on the same frozen embeddings
(so differences isolate the *clustering*, not the eval):

```
method                     test_frac pure_frac  entropy  test_acc  acc_drop
---------------------------------------------------------------------------
kmeans                         0.405     0.000    1.067     0.253    -0.641
ward                           0.485     0.000    1.286     0.516    -0.368
deepcluster-lite               0.400     0.000    1.053     0.351    -0.544
deepcluster-faithful           0.064     0.000    0.202     0.423    -0.448
kmeans@ft_cls                  0.007     0.000    0.042     0.302    -0.651
ward@ft_cls                    0.008     0.100    0.042     0.348    -0.609
```

**Semi-faithful (clustering #1 = frozen embeddings, one fine-tune total):** with
frozen text features, `kmeans`/`ward`/`deepcluster-lite` all land at similarly high
`label_entropy` (~1.0–1.3) — clusters are already label-diverse. DEEP CLUSTER's
diversification step has nothing to fix here; it doesn't beat the cheap surrogates.
This alone is inconclusive, because frozen features never produce the homogeneous
clusters the paper's method exists to break up.

**Faithful (`--task-finetune`, clustering #1 = task-fine-tuned `[CLS]`, two
fine-tunes) — the decisive test.** `kmeans@ft_cls` / `ward@ft_cls` cluster the
task-fine-tuned representation *directly*, with no DEEP CLUSTER step — this is the
"do it naively" control. Read `label_entropy` and `test_frac`, not `pure_frac`
(which requires *every* point in a cluster to share one label — too harsh a bar to
be informative here):

- **The paper's premise reproduces.** `label_entropy` collapses from ~1.0–1.3
  (frozen features) to **0.042** (task-fine-tuned `[CLS]`, plain clustering) — a
  >25x drop. Clusters become almost entirely single-label (a handful of stray
  points keep `pure_frac` at 0, but the near-zero entropy shows they're
  overwhelmingly homogeneous). `test_frac` correspondingly collapses to **0.7–0.8%**
  (~42–48 of 5952 examples): with near-pure clusters, `minority_route`'s
  majority-label→train / minority-label→test rule sends almost everything to train.
  This is the same degenerate-minority-set failure mode already documented for
  tabular data and for CLINC (footnote-10 flood), now confirmed on *genuinely*
  task-fine-tuned text features — not an artifact of the semi-faithful shortcut.
- **DEEP CLUSTER partially reverses it, as designed, but doesn't fully restore
  diversity.** `deepcluster-faithful` (pseudo-label fine-tune + recluster on top of
  the same task-fine-tuned `[CLS]`) lifts entropy 0.042 → **0.202** (~5x) and
  `test_frac` 0.007 → **0.064** (~9x) relative to the naive `kmeans@ft_cls`
  control. Directionally exactly what the paper claims. But it lands well short of
  the frozen-feature diversity level (entropy ~0.2 vs ~1.0–1.3, test_frac ~6% vs
  ~40–48%) — one deep-cluster iteration narrows the gap, it doesn't close it.
- **Caveat:** the `@ft_cls` and `deepcluster-faithful` test sets are tiny
  (~42–380 examples out of 5952), so their `test_acc`/`acc_drop` numbers are
  high-variance and not directly comparable in magnitude to the frozen-feature
  rows' `acc_drop`; treat `test_frac`/`label_entropy` as the reliable signal here,
  accuracy as secondary.

**Bottom line:** the paper's justification for DEEP CLUSTER over plain clustering
is real and reproduces under a faithful setup — task-fine-tuned features do
collapse to near label-pure clusters under naive clustering, and DEEP CLUSTER
measurably (if partially) undoes that. Our cheap in-library `deepcluster-lite`
surrogate solves a related but different problem: it was tuned against a
heavily-underfit pseudo-label MLP on top of *frozen* features, a regime where this
experiment shows there's no collapse to fix in the first place. The two backends
aren't interchangeable — `deepcluster-lite` targets the frozen-feature setting most
`splytters` users are actually in; a faithful task-fine-tuned pass is a heavier,
different tool for when the encoder itself is being fine-tuned.
