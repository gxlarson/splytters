"""Shared type aliases for splytters."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# A splitter takes an embeddings array (plus ``train_size`` / ``random_state``
# keyword arguments) and returns a ``(train_indices, test_indices)`` pair of
# integer ndarrays. Use this to annotate functions that accept a custom
# splitter (e.g. ``SplytterSplit``, ``splytter_train_test_split``, the interop
# helpers).
Splitter = Callable[..., tuple[np.ndarray, np.ndarray]]

__all__ = ["Splitter"]
