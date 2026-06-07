"""Shared pytest configuration.

* Seeds numpy/`random` before every test for reproducibility.
* Skips heavy-modality sorter test modules when their optional dependency is
  absent, so a *core-only* install (`pip install -e .`) collects and runs
  cleanly instead of erroring during import.
"""

from __future__ import annotations

import importlib.util
import random

import numpy as np
import pytest

# test-module filename -> optional dependency it requires to even import.
_OPTIONAL_TEST_DEPS = {
    "test_audio_sorters.py": "librosa",
    "test_image_sorters.py": "PIL",
    "test_text_sorters.py": "torch",
    "test_embedders.py": "sentence_transformers",
    "test_tabular_sorters.py": "pandas",
}

# Paths are relative to this conftest (repo root); the tests live in tests/.
collect_ignore = [
    f"tests/{fname}"
    for fname, dep in _OPTIONAL_TEST_DEPS.items()
    if importlib.util.find_spec(dep) is None
]


@pytest.fixture(autouse=True)
def _seed_everything():
    """Make every test deterministic regardless of execution order."""
    random.seed(0)
    np.random.seed(0)
    yield
