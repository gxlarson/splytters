# Prior-Work Faithfulness Audit

Date: 2026-07-09

Scope: first-pass audit of the strongest prior-work claims in `splytters.adversarial`
against the cited papers.

## Findings

### 1. `cluster_split(strategy="subset_sum" / "closest")` overclaims faithfulness

Location: `splytters/adversarial.py`, `cluster_split` references section.

Sources:

- Paper: https://aclanthology.org/2023.genbench-1.9/
- PDF: https://aclanthology.org/2023.genbench-1.9.pdf
- Paper-listed code: https://github.com/MaikeZuefle/Latent-Feature-Splits

The docstring says the `"subset_sum"` and `"closest"` strategies implement
SUBSET-SUM-SPLIT and CLOSEST-SPLIT from Züfle, Dankers, and Titov (2023),
"Latent Feature-based Data Splits to Improve Generalisation Evaluation."

The paper's method:

- clusters task-model hidden representations with k-means;
- enforces a fixed test-set size and equal train/test class distributions;
- runs the split search for `k = 3..50`;
- for SUBSET-SUM-SPLIT, solves/approximates a multidimensional subset-sum target
  over class-count vectors;
- for CLOSEST-SPLIT, selects an isolated cluster, grows a connected test region by
  nearest-neighbor clustering over centroids, then fills remaining test slots with
  individual examples nearest to test centroids.

Current implementation:

- uses the caller's fixed `n_clusters`;
- uses greedy subset selection for `"subset_sum"`;
- uses a centroid-distance pocket heuristic for `"closest"`;
- does not perform the paper's `k = 3..50` search;
- does not do the paper's final individual-example fill step;
- supports generic embeddings rather than requiring task-finetuned hidden states.

Assessment: useful approximation, but not faithful enough for "implements."

Recommended wording: "approximates" or "inspired by" Züfle et al. rather than
"implements."

### 2. `cluster_kfold` is not a faithful ClusterDataSplit / SDS K-means implementation

Location: `splytters/adversarial.py`, `cluster_kfold` references section.

Sources:

- Paper: https://aclanthology.org/2020.eval4nlp-1.15/
- PDF: https://aclanthology.org/2020.eval4nlp-1.15.pdf
- Paper-listed code: https://github.com/boschresearch/clusterdatasplit_eval4nlp-2020

The docstring says it implements the challenging clustering-based
cross-validation of Wecker, Friedrich, and Adel (2020), "ClusterDataSplit."

The paper's core algorithm is SDS K-means: Size and Distribution Sensitive
K-means. It modifies clustering itself so clusters/folds have approximately equal
size and controlled label distributions. The assignment/update procedure operates
with label-specific capacities and 1-on-1 swaps.

Current implementation:

- clusters with ordinary KMeans or DBSCAN;
- computes per-cluster class-count vectors after clustering;
- greedily assigns whole clusters to folds to reduce label-distribution overshoot;
- does not implement SDS K-means, same-size K-means, label-specific cluster
  capacities, or the swap-based update procedure.

Assessment: useful cluster-coherent, label-aware fold assignment heuristic, but
not a faithful ClusterDataSplit implementation.

Recommended wording: describe it as a lightweight heuristic inspired by
ClusterDataSplit, or implement SDS K-means in a separate PR if exact faithfulness
is important.

### 3. MMD splitters are mostly accurately qualified

Locations:

- `splytters/adversarial.py`, `mmd_maximized_split`
- `splytters/balanced.py`, `mmd_minimized_split`

Sources:

- OpenReview: https://openreview.net/forum?id=Q692C0WtiD
- arXiv: https://arxiv.org/abs/2405.19461
- arXiv PDF: https://arxiv.org/pdf/2405.19461
- Paper-listed code: none found in the accessible paper text.

The `mmd_maximized_split` docstring says it implements the max-MMD objective from
Napoli and White (2025), while explicitly noting that it does not implement the
paper's full constrained kernel-k-means method with linear programming.

The paper's method:

- proposes maximizing distribution mismatch between train and validation;
- uses MMD as the mismatch measure;
- shows the partitioning problem reduces to kernel k-means;
- adds constrained clustering with linear programming to control size, label, and
  optional group distributions.

Current implementation:

- directly optimizes the MMD objective by random-start swap optimization;
- does not implement kernel k-means;
- does not implement the LP constraints;
- does not preserve label/group balance except through whatever caller wraps
  around it.

Assessment: the docstring is mostly honest. The safest language is
"swap-optimized approximation of the max-MMD objective" rather than implying the
published algorithm is implemented.

## Recommended PR (first pass; superseded by "Updated recommended PR sequence" below)

Start with a documentation-accuracy PR:

- Change `cluster_split` prior-work wording from "implements" to "approximates"
  or "inspired by."
- Change `cluster_kfold` wording to avoid claiming faithful ClusterDataSplit /
  SDS K-means implementation.
- Keep MMD language qualified; optionally add "swap-optimized approximation."

Separate, larger algorithm PRs could then add faithful versions of:

- Züfle et al. `k = 3..50` search and final fill behavior;
- Wecker et al. SDS K-means;
- Napoli and White constrained kernel-k-means / LP splitting.

## Additional cited-method audit

### 4. `minority_split` is appropriately caveated

Location: `splytters/adversarial.py`, `minority_split` references section.

Sources:

- Paper: https://aclanthology.org/2023.findings-acl.833/
- PDF: https://aclanthology.org/2023.findings-acl.833.pdf
- Paper-listed code/data: https://github.com/schwartz-lab-NLP/fight-bias-with-bias

The function references Reif and Schwartz (2023), "Fighting Bias with Bias."
Their minority-examples method:

- clusters `[CLS]` representations from a model trained on the dataset;
- defines each cluster's majority label as biased;
- defines all non-majority labels in that cluster as anti-biased / minority
  examples;
- assigns test instances to the nearest training cluster;
- uses DeepCluster because standard clustering tends to produce label-homogeneous
  clusters.

Current implementation:

- implements the label-routing rule over clusters;
- exposes `minority_route` for externally supplied cluster labels;
- has `kmeans`, `ward`, and `deepcluster-lite` modes;
- explicitly warns that standard k-means is the clusterer the paper rejects;
- explicitly states that `deepcluster-lite` is a static-feature stand-in rather
  than faithful DeepCluster.

Assessment: the docstring is unusually honest and should remain as-is. The
implementation is faithful to the minority-label routing, but not to the paper's
full representation-learning setup unless users provide faithful DeepCluster
labels externally.

### 5. `minority_grow_split` is clearly an extension, not a paper algorithm

Location: `splytters/adversarial.py`, `minority_grow_split` references section.

Sources:

- Paper: https://aclanthology.org/2023.findings-acl.833/
- PDF: https://aclanthology.org/2023.findings-acl.833.pdf
- Paper-listed code/data: https://github.com/schwartz-lab-NLP/fight-bias-with-bias

The docstring says it extends the bias-amplified minority seed of Reif and
Schwartz. That is accurate: the proximity-growth behavior is a library addition,
not claimed as the paper's algorithm.

Assessment: no issue. Keep the "extends" wording.

### 6. Boundary splitters are framed as adaptations, not faithful implementations

Locations:

- `splytters/adversarial.py`, `class_boundary_split`
- `splytters/adversarial.py`, `decision_boundary_split`

Sources:

- Paper: https://aclanthology.org/2021.eacl-main.156/
- PDF: https://aclanthology.org/2021.eacl-main.156.pdf
- Paper-listed code: https://github.com/google-research/google-research/tree/master/talk_about_random_splits

The relevant prior work in Søgaard et al. (2021), "We Need to Talk About Random
Splits," motivates biased/adversarial splits and includes examples such as
training on short sentences and evaluating on long ones. The paper also
constructs adversarial splits by approximately maximizing Wasserstein distance
between splits.

Current implementation:

- `class_boundary_split` uses cross-class embedding geometry to hold out
  boundary/confusable examples;
- `decision_boundary_split` uses learned model margins or entropy;
- both are label-aware variants of the hard-split idea, not faithful algorithms
  from Søgaard et al.

Assessment: the docstrings already say "variant" and describe what differs.
No major issue.

### 7. `wasserstein_adversarial_split` is close to the cited Søgaard et al. algorithm

Location: `splytters/adversarial.py`, `wasserstein_adversarial_split`.

Sources:

- Paper: https://aclanthology.org/2021.eacl-main.156/
- PDF: https://aclanthology.org/2021.eacl-main.156.pdf
- Paper-listed code: https://github.com/google-research/google-research/tree/master/talk_about_random_splits

Søgaard et al. describe approximate adversarial splits by:

- computing Wasserstein distances between data points;
- building/querying a BallTree;
- randomly selecting a centroid for the test split;
- using nearest neighbors of that centroid as the test split;
- repeating runs to estimate worst-case behavior.

Current implementation:

- uses `scipy.stats.wasserstein_distance` as a nearest-neighbor metric;
- builds a `NearestNeighbors(..., algorithm="ball_tree")` index;
- samples a random anchor in the embedding bounding box;
- uses the nearest `n_test` examples as test.

Assessment: close in spirit and mechanics. The main difference is that the
anchor is sampled from the bounding box rather than selected as an existing data
point/centroid, and the function returns one split rather than averaging over
runs. The wording "adapted from" is appropriate.

### 8. Length sorters are heuristic features, not paper algorithm implementations

Locations:

- `splytters/sorters/text_sorters.py`, `character_length`
- `splytters/sorters/text_sorters.py`, `tokens_length`

Sources:

- Sogaard et al. paper: https://aclanthology.org/2021.eacl-main.156/
- Sogaard et al. paper-listed code: https://github.com/google-research/google-research/tree/master/talk_about_random_splits
- Varis and Bojar paper: https://aclanthology.org/2021.emnlp-main.650/
- Varis and Bojar PDF: https://aclanthology.org/2021.emnlp-main.650.pdf
- Lake and Baroni arXiv: https://arxiv.org/abs/1711.00350
- Lake and Baroni paper-listed SCAN data: https://github.com/brendenlake/SCAN
- Lake and Baroni related CommAI environment: https://github.com/facebookresearch/CommAI-env
- Siegelmann and Sontag DOI: https://doi.org/10.1145/130385.130432

The docstrings cite length-based generalization work, including Søgaard et al.,
Varis and Bojar, Lake and Baroni, and earlier sequence-length generalization
references. The functions simply rank by character or token count; they do not
claim to reproduce a full benchmark construction.

Assessment: no faithfulness issue. The wording "splitting by length to test
generalization" and "token-count variant" is appropriately modest.

### 9. `perplexity_score` supports Likelihood Splits, but the default is not the
paper's preferred score

Location: `splytters/sorters/text_sorters.py`, `perplexity_score`.

Sources:

- Paper: https://aclanthology.org/2023.findings-eacl.71/
- PDF: https://aclanthology.org/2023.findings-eacl.71.pdf
- Paper-listed code: https://github.com/ameyagodbole/long-tail-likelihood-splits

Godbole and Jia (2023), "Benchmarking Long-tail Generalization with Likelihood
Splits," construct splits by assigning total log-likelihood scores with a
language model, putting lower-likelihood examples in evaluation, optionally
controlling for length by bucketing. They explicitly note that perplexity
normalizes for length and can over-correct toward short examples.

Current implementation:

- offers `scoring="log_likelihood"`, which matches the paper's preferred score;
- defaults to `scoring="perplexity"`;
- sorts by typicality so low-likelihood / high-perplexity examples can land in
  the tail when paired with a sorter split;
- does not implement the paper's length-bucketed control split;
- does not implement the paper's task-specific prompts or fold-trained scoring
  protocol.

Assessment: the docstring is mostly honest because it explicitly says to use
`scoring="log_likelihood"` to reproduce the paper's choice. If the package wants
the paper-faithful behavior to be the default, change the default to
`"log_likelihood"` in a major/minor release. Otherwise, keep the current API but
consider renaming docs/examples to avoid implying that default perplexity is the
faithful Likelihood Split.

### 10. Text diversity / n-gram Jaccard references are reasonably scoped

Locations:

- `splytters/distances.py`
- `splytters/metrics.py`, `diversity_text`

Sources:

- PDF: https://aclanthology.org/N19-1051.pdf
- Paper-listed code: none found in the accessible paper text.

The cited Larson et al. (2019) paper proposes outlier detection and diversity
analysis for dialog data. Relevant parts:

- rank examples by distance from a class/corpus mean embedding;
- use pairwise distance over text corpora as a diversity measure;
- discuss n-gram/Jaccard-style text distance in the analysis.

Current implementation:

- provides `ngram_jaccard_similarity` / distance;
- computes mean unordered pairwise text distance in `diversity_text`;
- does not claim to implement the full outlier-detection/data-collection
  pipeline from the paper.

Assessment: no major issue. These are utility-level implementations inspired by
the cited measures, not overclaimed algorithm reproductions.

### 11. Audio sorters are appropriately described as heuristic split features

Location: `splytters/sorters/audio_sorters.py`.

Sources:

- Paper: https://aclanthology.org/2023.eacl-main.10/
- PDF: https://aclanthology.org/2023.eacl-main.10.pdf
- Paper-listed code: none found in the accessible paper text.

The docstrings cite Liu, Spence, and Prud'hommeaux (2023), "Investigating data
partitioning strategies for crosslinguistic low-resource ASR evaluation." That
paper investigates multiple data partitioning strategies in very low-resource
ASR and finds duration and intensity comparatively predictive of WER variability,
while random splits can be more reliable under data sparsity.

Current implementation:

- exposes per-utterance feature sorters such as duration, mean amplitude, RMS
  energy, and pitch;
- does not claim to reproduce the full ASR evaluation protocol;
- includes the paper's caution that heuristic/adversarial splits may behave like
  random splits under sparse data.

Assessment: no issue. The docstrings are appropriately scoped as heuristic
features.

## Coverage

This first pass audits the 11 cited methods listed above (cluster_split,
cluster_kfold, the two MMD splitters, minority_split, minority_grow_split,
the two boundary splitters, wasserstein_adversarial_split, the length
sorters, perplexity_score, the text diversity utilities, and the audio
sorters). Cited methods elsewhere in the package that are not covered by
these findings have not yet been audited.

## Updated recommended PR sequence

1. Documentation-accuracy PR for the two clear overclaims:
   - `cluster_split(strategy="subset_sum"/"closest")`
   - `cluster_kfold`
2. Optional wording PR for likelihood splits:
   - clarify that `scoring="log_likelihood"` is the paper-faithful mode;
   - decide whether the default should remain `perplexity` for API continuity.
3. Larger algorithm PRs only if exact faithfulness is desired:
   - Züfle et al. `k=3..50` search and final fill behavior;
   - Wecker et al. SDS K-means;
   - paper-faithful likelihood split helper with length bucketing;
   - Napoli and White constrained kernel-k-means / LP splitting.

---

## Review by Claude Opus 4.8 — 2026-07-09

_This section was authored by **Claude Opus 4.8** (Anthropic, via Claude Code). I
cross-checked the audit above against the current `splytters` source. I verified the
**code-side** claims directly; I did **not** independently re-read the cited papers, so
the paper-internal descriptions above are taken on trust here (see suggestion 2)._

### Verdict

Accurate, well-scoped, and mergeable as a notes artifact. Every code-side claim I
spot-checked holds up, and the two headline "overclaim" findings are real and actionable.

### Verified against the code

- **Finding 1 (`cluster_split`)** — confirmed. The docstring in `splytters/adversarial.py`
  literally says the `"subset_sum"`/`"closest"` strategies *"implement SUBSET-SUM-SPLIT and
  CLOSEST-SPLIT from Züfle…"*, while the code uses a caller-fixed `n_clusters`, greedy subset
  selection, and a centroid-distance pocket — no `k = 3..50` search and no individual-example
  fill step. The recommended "implements → approximates / inspired by" is the correct fix.
- **Finding 2 (`cluster_kfold`)** — confirmed. Docstring says *"Implements the challenging
  clustering-based cross-validation of Wecker…"*, but the code is ordinary KMeans/DBSCAN plus
  greedy whole-cluster fold assignment — not SDS K-means, label-specific capacities, or the
  swap-based update. Overclaim is real.
- **Finding 4 (`minority_split`)** — confirmed "unusually honest": the docstring explicitly
  calls `deepcluster-lite` *"a light surrogate for the paper's DEEP CLUSTER"* and names k-means
  as *"the clusterer the paper rejects."* Keep as-is.
- **Finding 9 (`perplexity_score`)** — confirmed: default is `scoring="perplexity"` with a
  `"log_likelihood"` option available, exactly as described.

### Suggestions

1. **Reconcile the MMD citation year.** Finding 3 cites *"Napoli and White (2025)"* but lists
   `arXiv:2405.19461`, a **May 2024** identifier. Likely a venue-vs-arXiv year difference — but
   in a *faithfulness* document, double-check the **author names and year** against the
   OpenReview/venue record. Author/year is the easiest detail to get wrong and the most
   conspicuous in this particular doc.
2. **State the audit's own provenance.** The doc confidently describes each paper's internal
   algorithm (Züfle's `k = 3..50` search, Wecker's label-capacity swaps, Napoli–White's kernel
   k-means + LP). Add one line on how those were established — read from the linked PDF vs
   inferred from the abstract — since the doc's credibility rests on its sources being checked.
3. **Remove the duplicate "Recommended PR" block.** The interim one (after Finding 3) is
   superseded by "Updated recommended PR sequence" at the end and reads as contradictory;
   delete it or explicitly label it "partial (first pass)".
4. **Make coverage explicit.** It says "first-pass." Add a short **"Methods not yet audited"**
   list so a reader knows the boundary of this pass (is every cited splitter/sorter covered, or
   just these 11?).
5. **Make the doc-fix PR trivial.** For Findings 1 & 2, quote the exact current docstring
   sentence and the proposed replacement inline, so the follow-up PR is copy-paste.
6. **File location.** Consider `docs/` or `notes/` rather than the repo root to keep the top
   level clean (minor).

### Bottom line

Approve the direction. The doc is honest and its two actionable findings check out. Merge
(ideally after fixing suggestion 1 and removing the duplicate block in suggestion 3), then do
the small docstring-accuracy PR it recommends for `cluster_split` and `cluster_kfold`.

---

## Review notes (Claude Fable 5, 2026-07-09)

This audit was independently verified by Claude Fable 5: every "Current
implementation" claim was checked against the repository source, and every
paper claim (URLs, authors, years, method descriptions) was checked against
the papers and repositories themselves. Overall verdict: accurate. One
factual error and two precision caveats were found.

### Corrections

- Finding 2 repo URL: `clusterdatasplit_eval4nlp-2020156` returns a 404. The
  correct URL is https://github.com/boschresearch/clusterdatasplit_eval4nlp-2020
  (archived May 2024). The stray "156" is a PDF copy-paste artifact: the
  paper's footnote URL is line-wrapped and immediately followed by the page
  number 156.

### Precision caveats

- Finding 3: "Napoli and White (2025)" refers to the TMLR publication year;
  the arXiv preprint (2405.19461) is from May 2024, and its v1 carried the
  subtitle "...for Domain Generalisation" before the camera-ready title
  change. The "no paper-listed code" claim was confirmed: no repository
  appears in the paper text or via search.
- Finding 10: the code cites Larson et al. (2019) only by ACL URL and figure
  number; the author names do not appear in the docstrings.

### Confirmations

- The two headline overclaims are real: `cluster_split` (adversarial.py:228)
  says the strategies "implement" SUBSET-SUM-SPLIT and CLOSEST-SPLIT, and
  `cluster_kfold` (adversarial.py:396) says "Implements" Wecker et al.
  (2020), while both bodies are the simplified heuristics this audit
  describes (fixed `n_clusters` with no `k = 3..50` search, greedy
  assignment, no per-example fill; ordinary KMeans/DBSCAN with greedy
  whole-cluster fold packing, no SDS K-means, capacities, or swaps).
- The paper method descriptions in findings 1, 2, 3, 4, 5, 7, and 9 match
  the papers' full texts, including the fine details (Züfle et al.'s
  multidimensional subset-sum over class-count vectors and closest-split
  individual-example fill; Wecker et al.'s 1-on-1 swap update; Søgaard et
  al.'s BallTree / random-centroid procedure; Godbole and Jia's footnote
  that perplexity over-corrects toward short examples).
- The "honest docstring" assessments (findings 3-11) are correct, and the
  deviations noted for `wasserstein_adversarial_split` (bounding-box anchor
  rather than a data-point centroid; single split rather than repeated runs)
  match the code exactly.
- All other GitHub repositories, ACL Anthology links, arXiv IDs, and the
  Siegelmann and Sontag DOI resolve to the claimed resources.

Note: this section also resolves suggestion 1 from the Opus 4.8 review above
(the Napoli and White year is the TMLR 2025 publication; the arXiv preprint
is 2024).
