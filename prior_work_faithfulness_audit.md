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
- Paper-listed code: https://github.com/boschresearch/clusterdatasplit_eval4nlp-2020156

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

## Recommended PR

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
