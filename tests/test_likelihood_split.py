"""Tests for the likelihood_split protocol (Godbole & Jia, 2023).

All tests use precomputed ``scores`` (and ``lengths`` where bucketing is
tested). The ``texts`` path would score with GPT-2 via ``perplexity_score``,
which cannot run in this environment (torch 2.1 vs transformers' torch>=2.4), so
no model is ever loaded here.
"""

import numpy as np
import pytest

from splytters import likelihood_split


def test_lowest_likelihood_goes_to_test():
    # Higher score == more likely == training-preferred.
    scores = [10.0, 1.0, 8.0, 2.0, 9.0]
    train, test = likelihood_split(scores=scores, train_size=0.6)
    # 60% of 5 -> 3 train (highest scores: idx 0,2,4); 2 test (lowest: idx 1,3).
    assert sorted(train.tolist()) == [0, 2, 4]
    assert sorted(test.tolist()) == [1, 3]


def test_exact_split_sizes_fraction():
    scores = np.linspace(0, 1, 10)
    train, test = likelihood_split(scores=scores, train_size=0.7)
    assert len(train) == 7
    assert len(test) == 3
    # Partition is complete and disjoint.
    assert set(train.tolist()) | set(test.tolist()) == set(range(10))
    assert not (set(train.tolist()) & set(test.tolist()))


def test_exact_split_sizes_absolute_count():
    scores = np.arange(8, dtype=float)
    train, test = likelihood_split(scores=scores, train_size=5)
    assert len(train) == 5
    assert len(test) == 3
    # Absolute train count keeps the 5 highest scores.
    assert sorted(train.tolist()) == [3, 4, 5, 6, 7]
    assert sorted(test.tolist()) == [0, 1, 2]


def test_ties_broken_by_index_deterministically():
    scores = [5.0, 5.0, 5.0, 5.0]
    train, test = likelihood_split(scores=scores, train_size=0.5)
    # All tied -> lowest indices land in test (ascending index tiebreak).
    assert sorted(test.tolist()) == [0, 1]
    assert sorted(train.tolist()) == [2, 3]


def test_determinism_repeated_calls_match():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=50)
    a = likelihood_split(scores=scores, train_size=0.7)
    b = likelihood_split(scores=scores, train_size=0.7)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_length_bucketing_differs_from_global_split():
    # Construct a case where low likelihood correlates with long length:
    # long examples (idx 4-7) all have the lowest scores. A global split sends
    # all long examples to test; a length-bucketed split must instead take the
    # lowest scorer from EACH length bucket, mixing short and long into test.
    lengths = [1, 2, 3, 4, 10, 11, 12, 13]
    #          short group (0-3)     long group (4-7)
    scores = [4.0, 3.0, 2.0, 1.0, 0.4, 0.3, 0.2, 0.1]

    # Global: lowest 2 scores overall are the two longest examples.
    g_train, g_test = likelihood_split(scores=scores, train_size=0.75)
    assert sorted(g_test.tolist()) == [6, 7]

    # Bucketed (2 buckets by length): each bucket contributes its lowest scorer.
    b_train, b_test = likelihood_split(
        scores=scores, lengths=lengths, length_buckets=2, train_size=0.75
    )
    # Short bucket {0,1,2,3} lowest score -> idx 3; long bucket {4,5,6,7} -> idx 7.
    assert sorted(b_test.tolist()) == [3, 7]
    # The two splits genuinely differ.
    assert sorted(b_test.tolist()) != sorted(g_test.tolist())


def test_length_bucketing_pulls_low_scorers_from_every_bucket():
    # Three length buckets; within each, the single lowest scorer is eval.
    lengths = [1, 1, 5, 5, 9, 9]
    scores = [2.0, 1.0, 2.0, 1.0, 2.0, 1.0]  # odd indices are the low scorers
    train, test = likelihood_split(
        scores=scores, lengths=lengths, length_buckets=3, train_size=0.5
    )
    # 50% train -> one train + one test per 2-element bucket: low scorer (odd) tests.
    assert sorted(test.tolist()) == [1, 3, 5]
    assert sorted(train.tolist()) == [0, 2, 4]


def test_texts_path_matches_equivalent_scores(monkeypatch):
    """The ``texts`` path scores via ``perplexity_score`` (imported lazily inside
    ``_resolve_scores``) and reindexes its ranked ``[(idx, score), ...]`` output
    back to positional order. Mock the scorer -- transformers/torch cannot run
    here -- with a deliberately shuffled ranking, and assert the resulting split
    is identical to the equivalent precomputed ``scores=`` call.

    Patches the name on ``splytters.sorters.text_sorters`` because the helper
    does ``from splytters.sorters.text_sorters import perplexity_score`` at call
    time, so that module attribute is the lookup target.
    """
    import splytters.sorters.text_sorters as ts

    texts = ["a", "b", "c", "d", "e"]
    # Positional (per-example) total log-likelihoods; higher == more likely.
    positional = [10.0, 1.0, 8.0, 2.0, 9.0]
    # perplexity_score returns (index, score) tuples in ranked (non-positional)
    # order -- here ascending by score -- which _resolve_scores must scatter back
    # to positional order before splitting.
    ranked = [(1, 1.0), (3, 2.0), (2, 8.0), (4, 9.0), (0, 10.0)]

    def fake_perplexity_score(text_list, model=None, tokenizer=None, scoring=None):
        assert list(text_list) == texts
        assert scoring == "log_likelihood"  # the paper's score, never perplexity
        return ranked

    monkeypatch.setattr(ts, "perplexity_score", fake_perplexity_score)

    t_train, t_test = likelihood_split(texts=texts, train_size=0.6)
    s_train, s_test = likelihood_split(scores=positional, train_size=0.6)
    assert np.array_equal(t_train, s_train)
    assert np.array_equal(t_test, s_test)
    # Lowest-likelihood examples (idx 1, 3) land in test, matching the scores path.
    assert sorted(t_test.tolist()) == [1, 3]


def test_texts_path_length_buckets_uses_char_length(monkeypatch):
    """With ``length_buckets`` and no explicit ``lengths``, the texts branch of
    ``_resolve_lengths`` measures character length of the texts. Feed texts whose
    char lengths reproduce the bucketing test's lengths and assert the split
    matches the equivalent ``scores=`` + explicit ``lengths=`` call.
    """
    import splytters.sorters.text_sorters as ts

    # Char lengths 1,2,3,4 (short bucket) and 10,11,12,13 (long bucket).
    texts = [
        "a", "bb", "ccc", "dddd",
        "x" * 10, "x" * 11, "x" * 12, "x" * 13,
    ]
    positional = [4.0, 3.0, 2.0, 1.0, 0.4, 0.3, 0.2, 0.1]
    # Shuffled ranking (ascending by score) to also exercise the reindex.
    ranked = sorted(enumerate(positional), key=lambda p: p[1])

    def fake_perplexity_score(text_list, model=None, tokenizer=None, scoring=None):
        assert scoring == "log_likelihood"
        return ranked

    monkeypatch.setattr(ts, "perplexity_score", fake_perplexity_score)

    t_train, t_test = likelihood_split(
        texts=texts, length_buckets=2, train_size=0.75
    )
    s_train, s_test = likelihood_split(
        scores=positional, lengths=[len(t) for t in texts],
        length_buckets=2, train_size=0.75,
    )
    assert np.array_equal(t_train, s_train)
    assert np.array_equal(t_test, s_test)
    # Each length bucket contributes its lowest scorer: idx 3 (short), 7 (long).
    assert sorted(t_test.tolist()) == [3, 7]


def test_bucketing_with_scores_and_no_lengths_raises():
    with pytest.raises(ValueError, match="lengths"):
        likelihood_split(scores=[1.0, 2.0, 3.0, 4.0], length_buckets=2, train_size=0.5)


def test_neither_texts_nor_scores_raises():
    with pytest.raises(ValueError, match="exactly one"):
        likelihood_split(train_size=0.5)


def test_both_texts_and_scores_raises():
    with pytest.raises(ValueError, match="exactly one"):
        likelihood_split(texts=["a", "b"], scores=[1.0, 2.0], train_size=0.5)


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.2])
def test_bad_fractional_train_size_raises(bad):
    with pytest.raises(ValueError, match="train_size"):
        likelihood_split(scores=[1.0, 2.0, 3.0, 4.0], train_size=bad)


@pytest.mark.parametrize("bad", [0, 4, 10])
def test_bad_absolute_train_size_raises(bad):
    with pytest.raises(ValueError, match="train_size"):
        likelihood_split(scores=[1.0, 2.0, 3.0, 4.0], train_size=bad)


def test_too_few_samples_raises():
    with pytest.raises(ValueError, match="at least 2"):
        likelihood_split(scores=[1.0], train_size=0.5)


def test_bad_length_buckets_raises():
    with pytest.raises(ValueError, match="length_buckets"):
        likelihood_split(
            scores=[1.0, 2.0, 3.0, 4.0], lengths=[1, 2, 3, 4],
            length_buckets=0, train_size=0.5,
        )


def test_lengths_wrong_size_raises():
    with pytest.raises(ValueError, match="lengths"):
        likelihood_split(
            scores=[1.0, 2.0, 3.0, 4.0], lengths=[1, 2, 3],
            length_buckets=2, train_size=0.5,
        )
