# Prior-Work Faithfulness: Fix Plan

Date: 2026-07-09
Author: Claude Fable 5, from the findings in `prior_work_faithfulness_audit.md`
(the audit plus the Opus 4.8 and Fable 5 review sections).

Guiding constraint: never replace an existing implementation. Where the code
diverges from a cited paper, the paper-faithful algorithm is added as an
opt-in mode and the current behavior remains the default.

## Analysis

The audit surfaces three distinct kinds of problems:

1. **Errors in the audit document itself** -- a broken repository URL, a
   duplicate recommendations block, and a missing coverage statement. Pure
   doc cleanup, zero risk.
2. **Docstring overclaims in the code** -- two hard overclaims
   (`cluster_split` says "implement", `cluster_kfold` says "Implements")
   and three soft precision issues (MMD wording, `perplexity_score`
   default vs. paper-faithful mode, Larson et al. cited without author
   names). Wording-only changes, no behavior change.
3. **Genuine algorithm gaps vs. the papers** -- the code is a simplified
   heuristic where the paper specifies a richer procedure. Each gap fits
   naturally as an additive mode on the existing API:
   - **Zuefle et al. 2023** (`cluster_split`): paper searches k = 3..50 and,
     for CLOSEST-SPLIT, fills remaining test slots with individual nearest
     examples. `cluster_split` already takes `n_clusters: int = 10`; extend
     `n_clusters` to accept a range / `"search"` sentinel and add
     `fill_individual: bool = False`. Defaults unchanged.
   - **Wecker et al. 2020** (`cluster_kfold`): paper's core is SDS K-means
     (size- and distribution-sensitive clustering with label-specific
     capacities and 1-on-1 swaps). `cluster_kfold` already dispatches on
     `method: str = "kmeans"`; add `method="sds_kmeans"` alongside the
     existing kmeans/dbscan + greedy fold packing.
   - **Napoli and White (TMLR 2025; arXiv 2024)** (`mmd_maximized_split`,
     `mmd_minimized_split`): paper's full method is constrained kernel
     k-means solved with linear programming (size/label/group constraints).
     Add a `method="kernel_kmeans"` (or `solver=`) parameter with optional
     `y`/`groups`; the random-start swap optimizer stays the default.
     `scipy.optimize.linprog` is already available via the scipy dependency.
   - **Godbole and Jia 2023** (`perplexity_score`): paper's protocol is
     total log-likelihood with a length-bucketed control split. Keep
     `scoring="perplexity"` as the default for API continuity and add the
     missing piece as a new splitter-level helper (e.g. `likelihood_split`
     with a `length_buckets` option) rather than changing the sorter.

## PR sequence

1. **Audit-doc cleanup** (can land on the PR #47 branch before merge):
   - fix the finding-2 repo URL: `clusterdatasplit_eval4nlp-2020156` ->
     https://github.com/boschresearch/clusterdatasplit_eval4nlp-2020;
   - delete or label the interim "Recommended PR" block (after finding 3)
     that is superseded by "Updated recommended PR sequence";
   - add a short "coverage / methods not yet audited" note.

2. **Docstring accuracy** (wording only, mergeable immediately):
   - `cluster_split` (adversarial.py:228): "implement" -> "adapts /
     approximates ... (deviations: fixed n_clusters, greedy selection, no
     k = 3..50 search, no individual-example fill)";
   - `cluster_kfold` (adversarial.py:396): "Implements" -> "A
     cluster-coherent, label-aware fold heuristic inspired by
     ClusterDataSplit (does not implement SDS K-means)";
   - MMD docstrings: add "swap-optimized approximation of the max-MMD
     objective" and cite "TMLR 2025; arXiv 2024";
   - add Larson et al. author names next to the N19-1051 URL citations in
     `distances.py` and `metrics.py`.

3. **Faithful Zuefle mode for `cluster_split`**: k-search over a cluster
   range plus individual-example fill, behind new defaults-off parameters;
   docstring describes both modes; tests compare fixed-k vs. searched-k.

4. **SDS K-means for `cluster_kfold`** as `method="sds_kmeans"`: the
   largest single implementation (capacity-constrained assignment plus
   swap-based updates). Reference: the archived Bosch repo above.

5. **Constrained kernel k-means / LP for the MMD splitters**: new method
   option on both `mmd_maximized_split` and `mmd_minimized_split` with
   optional label/group constraints via linear programming.

6. **Likelihood-split helper**: length-bucketed `likelihood_split` built on
   `perplexity_score(scoring="log_likelihood")`.

PRs 3-6 are independent and can land in any order. Each should also update
its docstring from "approximates" to "implements X when `<faithful option>`
is set, otherwise a lightweight approximation" -- resolving the audit's
findings without removing any existing behavior.

## Open decision

Whether `perplexity_score`'s default should ever flip to
`"log_likelihood"`. Recommendation: no -- keep the current default for API
continuity; the paper-faithful path becomes explicit via the new
`likelihood_split` helper.
