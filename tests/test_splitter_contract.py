"""Parametrized contract suite over every registered splitter.

``test_introspection`` only checks that each name in ``list_splitters()`` is
callable. This module asserts the *behavioural* contract every splitter must
honour, for the whole registry at once:

  (a) structural invariants -- integer-ndarray returns, disjoint, full cover, no
      duplicates, and (where the splitter has a ``train_size``) a train ratio
      within tolerance;
  (b) determinism -- two calls with the same ``random_state`` return identical
      indices;
  (c) it runs at a small ``n`` (n=8, train_size=0.5) without crashing.

Each splitter needs its own argument table entry in ``SPLITTER_SPECS`` below
(some require ``y`` or ``groups``, some need a small ``n_clusters`` to run at
n=8). ``test_registry_is_fully_specified`` fails loudly if a newly registered
splitter has no entry, forcing future additions to declare how to exercise them
here. Splitters that need an unavailable optional dependency raise ``ImportError``
from the call and are skipped with an explicit reason.
"""

from __future__ import annotations

import numpy as np
import pytest

import splytters

# ---------------------------------------------------------------------------
# Per-splitter argument table
# ---------------------------------------------------------------------------
# Each spec may set:
#   kwargs        extra keyword args passed on every call (beyond embeddings and
#                 the injected train_size / y / groups)
#   needs         extra positional-ish data the splitter requires: "y", "groups"
#   has_train_size  whether the splitter accepts train_size (default True)
#   ratio_tol     tolerance for the train-ratio invariant (default 0.15)
# ``n_clusters`` is pinned small so clustering splitters run at n=8.

_FAST = {"n_iterations": 100}

SPLITTER_SPECS: dict[str, dict] = {
    "random_split": {},
    "cluster_split": {"kwargs": {"n_clusters": 2}, "ratio_tol": 0.3},
    "centroid_adversarial_split": {"kwargs": {"n_clusters": 2}, "ratio_tol": 0.3},
    "distance_adversarial_split": {},
    "density_adversarial_split": {},
    "outlier_adversarial_split": {"ratio_tol": 0.2},
    "min_cut_split": {"kwargs": {"method": "spectral"}, "ratio_tol": 0.4},
    "normalized_cut_split": {"ratio_tol": 0.4},
    "wasserstein_adversarial_split": {"ratio_tol": 0.3},
    "mmd_maximized_split": {"kwargs": _FAST},
    "mmd_minimized_split": {"kwargs": _FAST},
    "minority_split": {
        "kwargs": {"n_clusters": 2},
        "needs": ["y"],
        "has_train_size": False,
    },
    "minority_grow_split": {"kwargs": {"n_clusters": 2}, "needs": ["y"]},
    "class_boundary_split": {"needs": ["y"]},
    "decision_boundary_split": {"kwargs": {"cv": 2}, "needs": ["y"], "ratio_tol": 0.2},
    "maximin_split": {},
    "cluster_leak_split": {"kwargs": {"n_clusters": 2}, "ratio_tol": 0.4},
    "neighbor_coverage_split": {"kwargs": {"k": 2}, "ratio_tol": 0.4},
    "centroid_matched_split": {"kwargs": {"n_iterations": 100}},
    "stratified_similarity_split": {"ratio_tol": 0.4},
    "nearest_neighbor_split": {"ratio_tol": 0.4},
    "duplicate_spread_split": {"ratio_tol": 0.4},
    "max_coverage_split": {"ratio_tol": 0.45},
    "distribution_matched_split": {"kwargs": _FAST},
    "moment_matched_split": {"kwargs": _FAST},
    "histogram_matched_split": {"kwargs": _FAST},
    "stratified_random_split": {"needs": ["y"]},
    "density_balanced_split": {"ratio_tol": 0.4},
    "group_split": {"needs": ["groups"], "ratio_tol": 0.4},
    "deduplicated_split": {"ratio_tol": 0.4},
}

SPLITTER_NAMES = splytters.list_splitters()


# ---------------------------------------------------------------------------
# Data + invocation helpers
# ---------------------------------------------------------------------------

def _dataset(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two well-separated Gaussian blobs with interleaved 2-class labels and
    small groups. Interleaved labels keep every blob/cluster label-impure so the
    label-driven splitters (minority_*) always find minority examples."""
    rng = np.random.RandomState(0)
    half = n // 2
    a = rng.randn(half, 2) + np.array([6, 6])
    b = rng.randn(n - half, 2) + np.array([-6, -6])
    X = np.vstack([a, b])
    y = np.array([i % 2 for i in range(n)])
    groups = np.array([i // 2 for i in range(n)])  # pairs -> whole-group splits
    return X, y, groups


def _call(name: str, n: int, *, train_size: float, random_state: int):
    """Invoke splitter ``name`` on an n-point dataset with the table's kwargs."""
    spec = SPLITTER_SPECS[name]
    fn = getattr(splytters, name)
    X, y, groups = _dataset(n)
    args: list = [X]
    for need in spec.get("needs", []):
        args.append(y if need == "y" else groups)
    kwargs = dict(spec.get("kwargs", {}))
    kwargs["random_state"] = random_state
    if spec.get("has_train_size", True):
        kwargs["train_size"] = train_size
    return fn(*args, **kwargs)


def _assert_valid_split(train, test, n_samples, *, train_size, ratio_tol,
                        check_ratio):
    """Structural invariants every split must satisfy (replicates the helper in
    test_splitters.assert_valid_split, plus an integer-dtype and a skippable
    ratio check for splitters whose test size is data-driven)."""
    assert isinstance(train, np.ndarray) and isinstance(test, np.ndarray)
    assert np.issubdtype(train.dtype, np.integer)
    assert np.issubdtype(test.dtype, np.integer)

    train_set, test_set = set(train.tolist()), set(test.tolist())
    assert len(train_set) == len(train), "duplicate indices in train"
    assert len(test_set) == len(test), "duplicate indices in test"
    assert not (train_set & test_set), "train/test overlap"
    assert train_set | test_set == set(range(n_samples)), "does not cover all indices"

    if check_ratio:
        ratio = len(train) / n_samples
        assert abs(ratio - train_size) < ratio_tol, (
            f"train ratio {ratio:.2f} outside {train_size} +/- {ratio_tol}"
        )


def _run(name, n, *, train_size, random_state):
    """Call the splitter, translating a missing optional dependency into a skip."""
    try:
        return _call(name, n, train_size=train_size, random_state=random_state)
    except ImportError as exc:  # optional backend (e.g. networkx) unavailable
        pytest.skip(f"{name} requires an unavailable optional dependency: {exc}")


# ---------------------------------------------------------------------------
# Registry coverage guard
# ---------------------------------------------------------------------------

def test_registry_is_fully_specified():
    """Every registered splitter must have a SPLITTER_SPECS entry (and vice
    versa). A new splitter added without an entry fails here with a clear
    message, forcing it to be registered in this contract suite."""
    registry = set(SPLITTER_NAMES)
    specified = set(SPLITTER_SPECS)
    missing = registry - specified
    stale = specified - registry
    assert not missing, (
        f"splitter(s) {sorted(missing)} are registered but have no "
        "SPLITTER_SPECS entry -- add one so the contract suite exercises them"
    )
    assert not stale, (
        f"SPLITTER_SPECS has entries for unregistered splitter(s) {sorted(stale)}"
    )


# ---------------------------------------------------------------------------
# The contract, parametrized over the whole registry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", SPLITTER_NAMES)
def test_produces_valid_split(name):
    spec = SPLITTER_SPECS[name]
    n = 40
    train, test = _run(name, n, train_size=0.7, random_state=0)
    _assert_valid_split(
        train, test, n,
        train_size=0.7,
        ratio_tol=spec.get("ratio_tol", 0.15),
        check_ratio=spec.get("has_train_size", True),
    )


@pytest.mark.parametrize("name", SPLITTER_NAMES)
def test_deterministic_for_fixed_random_state(name):
    a = _run(name, 40, train_size=0.7, random_state=0)
    b = _run(name, 40, train_size=0.7, random_state=0)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), (
        f"{name} is not deterministic for a fixed random_state"
    )


@pytest.mark.parametrize("name", SPLITTER_NAMES)
def test_runs_at_small_n(name):
    spec = SPLITTER_SPECS[name]
    n = 8
    train, test = _run(name, n, train_size=0.5, random_state=0)
    _assert_valid_split(
        train, test, n,
        train_size=0.5,
        ratio_tol=max(spec.get("ratio_tol", 0.15), 0.3),
        check_ratio=spec.get("has_train_size", True),
    )
