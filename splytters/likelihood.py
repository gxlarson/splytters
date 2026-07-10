"""
Likelihood Splits (Godbole & Jia, 2023).

This module reproduces the core split-construction protocol of

    Godbole & Jia (2023), "Benchmarking Long-tail Generalization with
    Likelihood Splits," Findings of the ACL: EACL, pp. 963-983.
    https://aclanthology.org/2023.findings-eacl.71

Every example is scored by the *total log-likelihood* a language model assigns
to it, and the lowest-likelihood examples become the evaluation set while the
higher-likelihood examples become training. An optional length-bucketed control
variant buckets examples by length and takes the lowest-likelihood fraction
*within each bucket*, which removes the confound that low-likelihood examples
tend to be long (the paper's "-len" splits).

The scoring itself reuses :func:`splytters.sorters.text_sorters.perplexity_score`
with ``scoring="log_likelihood"``. Note that ``perplexity_score``'s *default*
scoring is perplexity, which is length-normalized and is deliberately *not* the
paper's choice (perplexity over-corrects toward short examples in the tail; see
their fn. 3) -- this helper always uses total log-likelihood.

Scope vs. the paper. This reproduces the frozen-LM, promptless variant of the
paper's ``ll_split_pt``. It does not implement the paper's other protocol
elements: (a) a task-specific prompt prepended to each query, (b) scoring the
*query span only* (``perplexity_score`` scores the whole string you pass, so
pass the query text alone to approximate this), or (c) the ``ll_split`` k-fold
cross-fitted fine-tuned scorer. The length-bucket control uses whitespace token
count by default as a dependency-free stand-in for the paper's NLTK
``word_tokenize`` count (see :func:`likelihood_split`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from splytters.utils import apportion_train, as_index_array, resolve_n_train

if TYPE_CHECKING:  # only for annotations (no runtime import of transformers)
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def _resolve_scores(
    texts: Sequence[str] | None,
    scores: ArrayLike | None,
    model: GPT2LMHeadModel | None,
    tokenizer: GPT2TokenizerFast | None,
) -> np.ndarray:
    """Return a 1-D float array of per-example log-likelihood scores.

    Exactly one of ``texts`` / ``scores`` must be given. When ``texts`` is
    given, scores are computed with ``perplexity_score(scoring="log_likelihood")``
    (imported lazily so the optional transformers dependency is only required on
    that path).
    """
    if (texts is None) == (scores is None):
        raise ValueError(
            "pass exactly one of `texts` or `scores` "
            "(got both or neither)."
        )

    if scores is not None:
        arr = np.asarray(scores, dtype=float).ravel()
        return arr

    # texts path: score with the paper's total log-likelihood.
    # Imported lazily, matching how text_sorters keeps transformers optional.
    from splytters.sorters.text_sorters import perplexity_score

    ranked = perplexity_score(
        list(texts),  # type: ignore[arg-type]
        model=model,
        tokenizer=tokenizer,
        scoring="log_likelihood",
    )
    arr = np.empty(len(ranked), dtype=float)
    for idx, score in ranked:
        arr[idx] = score
    return arr


def _length_bucket_ids(
    lengths: np.ndarray, n_buckets: int
) -> list[np.ndarray]:
    """Bucket sample indices into ``n_buckets`` equal-count length buckets.

    Indices are ordered by ascending length and split into ``n_buckets``
    near-equal contiguous groups (quantile / equal-frequency bucketing). Ties in
    length are broken by index so the bucketing is deterministic.
    """
    order = np.lexsort((np.arange(len(lengths)), lengths))  # length, then index
    return [b for b in np.array_split(order, n_buckets) if len(b) > 0]


def likelihood_split(
    texts: Sequence[str] | None = None,
    scores: ArrayLike | None = None,
    train_size: float | int = 0.7,
    *,
    length_buckets: int | None = None,
    lengths: ArrayLike | None = None,
    model: GPT2LMHeadModel | None = None,
    tokenizer: GPT2TokenizerFast | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Likelihood Split of Godbole & Jia (2023).

    Scores every example by the *total log-likelihood* assigned by a language
    model and puts the lowest-likelihood ``1 - train_size`` fraction in the
    evaluation (test) set, leaving the higher-likelihood examples for training.
    This reproduces the paper's core protocol: "examples that are assigned lower
    likelihood by a pre-trained language model are placed in the test set, and
    more likely examples are in the training set."

    Scores may be supplied directly (``scores``) or computed from raw ``texts``
    via :func:`splytters.sorters.text_sorters.perplexity_score` with
    ``scoring="log_likelihood"`` (the paper's choice). Pass exactly one of the
    two. Note that ``perplexity_score``'s *default* is length-normalized
    perplexity, which is deliberately **not** the paper's score; this helper
    always requests total log-likelihood.

    Length-bucketed control ("-len" splits): with ``length_buckets=N``, examples
    are grouped into ``N`` equal-count buckets by length, and within each bucket
    the lowest-likelihood fraction is sent to evaluation. This removes the
    confound that low-likelihood examples tend to be long, so the train/eval
    length distributions match. Lengths come from ``lengths`` if given, otherwise
    from the whitespace token count of ``texts`` (``len(t.split())``) -- a
    dependency-free approximation of the paper's NLTK ``word_tokenize`` count.
    Pass ``lengths`` explicitly to reproduce the paper's tokenizer exactly.

    Args:
        texts: raw strings to score (mutually exclusive with ``scores``). Scored
            with total log-likelihood; requires the optional transformers/torch
            dependency of ``perplexity_score``.
        scores: precomputed per-example total log-likelihood scores, where
            *higher* means more likely (training-preferred). Mutually exclusive
            with ``texts``.
        train_size: fraction in the open interval (0, 1), or an absolute count in
            ``[1, n_samples)``, for the training set. Mirrors
            ``sklearn.model_selection.train_test_split``. The remaining
            lowest-likelihood examples form the evaluation set.
        length_buckets: if given, bucket examples into this many equal-count
            length buckets and take the lowest-likelihood fraction within each
            bucket (the paper's length-controlled variant). ``None`` (default)
            takes the globally lowest-likelihood examples.
        lengths: per-example lengths used for bucketing. Defaults to the
            whitespace token count of ``texts`` (``len(t.split())``), a
            dependency-free stand-in for the paper's NLTK token count; pass
            explicit ``lengths`` to match the paper's tokenizer. Required (with
            ``length_buckets``) when only ``scores`` are given.
        model: optional HuggingFace causal LM forwarded to ``perplexity_score``
            (defaults to GPT-2). Only used on the ``texts`` path.
        tokenizer: optional HuggingFace tokenizer forwarded to
            ``perplexity_score``. Only used on the ``texts`` path.

    Returns:
        ``(train_indices, test_indices)`` integer ndarrays, sorted ascending.
        ``test_indices`` are the lowest-likelihood examples.

    Raises:
        ValueError: if neither or both of ``texts``/``scores`` are given; if
            ``train_size`` is out of range; if there are fewer than 2 samples;
            if ``length_buckets`` is given without a way to determine lengths; or
            if ``lengths`` length does not match the number of samples.

    References:
        Godbole & Jia (2023), "Benchmarking Long-tail Generalization with
        Likelihood Splits," Findings of the ACL: EACL, pp. 963-983.
        https://aclanthology.org/2023.findings-eacl.71 . Reference code:
        https://github.com/ameyagodbole/long-tail-likelihood-splits . The paper
        scores examples with the total log-likelihood of the query tokens under
        GPT-2 and places the lowest-scoring examples in evaluation, optionally
        bucketing by length (NLTK ``word_tokenize`` count of the query) to
        control for the length confound. This helper reproduces that core
        protocol with a frozen, promptless GPT-2 scoring the whole passed string;
        it does not add the paper's task prompt, query-only scoring span, or
        k-fold cross-fitted fine-tuned scorer (``ll_split``). The length default
        approximates the NLTK count with a whitespace split; pass ``lengths`` to
        match the paper exactly.
    """
    score_arr = _resolve_scores(texts, scores, model, tokenizer)
    n_samples = len(score_arr)

    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples to split, got {n_samples}")

    is_int = isinstance(train_size, (int, np.integer)) and not isinstance(
        train_size, bool
    )
    if is_int:
        if not (1 <= train_size < n_samples):
            raise ValueError(
                f"train_size as an absolute count must be in [1, {n_samples}), "
                f"got {train_size}"
            )
    elif not (isinstance(train_size, (float, np.floating)) and 0 < train_size < 1):
        raise ValueError(
            "train_size as a fraction must be between 0 and 1 exclusive, "
            f"got {train_size!r}"
        )

    n_train = resolve_n_train(n_samples, train_size)

    if length_buckets is None:
        return _global_split(score_arr, n_train)

    if not (isinstance(length_buckets, (int, np.integer)) and length_buckets >= 1):
        raise ValueError(
            f"length_buckets must be a positive integer, got {length_buckets!r}"
        )

    length_arr = _resolve_lengths(lengths, texts, n_samples)
    return _bucketed_split(score_arr, length_arr, int(length_buckets), n_train)


def _resolve_lengths(
    lengths: ArrayLike | None, texts: Sequence[str] | None, n_samples: int
) -> np.ndarray:
    """Return a 1-D length array, from ``lengths`` or the token count of texts.

    Defaults to the whitespace token count (``len(t.split())``), a dependency-free
    approximation of the paper's NLTK ``word_tokenize`` count; callers wanting the
    exact tokenizer pass ``lengths``.
    """
    if lengths is not None:
        length_arr = np.asarray(lengths, dtype=float).ravel()
        if len(length_arr) != n_samples:
            raise ValueError(
                f"lengths has {len(length_arr)} entries but there are "
                f"{n_samples} samples"
            )
        return length_arr
    if texts is not None:
        return np.array([len(t.split()) for t in texts], dtype=float)
    raise ValueError(
        "length_buckets requires `lengths` when only `scores` are given "
        "(no texts to measure length from)."
    )


def _global_split(
    scores: np.ndarray, n_train: int
) -> tuple[np.ndarray, np.ndarray]:
    """Lowest-likelihood ``n - n_train`` examples to test, rest to train."""
    # Ascending by (score, index): lowest likelihood first, ties broken by index.
    order = np.lexsort((np.arange(len(scores)), scores))
    n_test = len(scores) - n_train
    test = order[:n_test]
    train = order[n_test:]
    return as_index_array(sorted(train.tolist())), as_index_array(sorted(test.tolist()))


def _bucketed_split(
    scores: np.ndarray, lengths: np.ndarray, n_buckets: int, n_train: int
) -> tuple[np.ndarray, np.ndarray]:
    """Within each length bucket, send the lowest-likelihood fraction to test."""
    buckets = _length_bucket_ids(lengths, n_buckets)
    # Distribute the global train budget across buckets proportionally so the
    # per-bucket train/eval fraction matches the requested split.
    per_bucket_train = apportion_train([len(b) for b in buckets], n_train)

    train: list[int] = []
    test: list[int] = []
    for bucket, n_train_b in zip(buckets, per_bucket_train, strict=True):
        # Ascending by (score, index) within the bucket: lowest likelihood first.
        order = bucket[np.lexsort((bucket, scores[bucket]))]
        n_test_b = len(bucket) - int(n_train_b)
        test.extend(order[:n_test_b].tolist())
        train.extend(order[n_test_b:].tolist())

    return as_index_array(sorted(train)), as_index_array(sorted(test))
