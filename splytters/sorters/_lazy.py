"""Lazy module proxy for the modality sorters.

Some modality sorters reference a heavy, optional dependency (librosa, Pillow,
pandas) pervasively across their functions. Importing it at module load would
make e.g. ``import splytters.sorters.audio_sorters`` require librosa even just
to read a docstring or list the sorters. Binding the dependency to a
``LazyModule`` proxy defers the import to first use, so importing a sorter
module stays dependency-free; only *calling* a sorter pulls the library in.

Usage::

    librosa = LazyModule("librosa")     # instead of: import librosa
    ...
    librosa.feature.mfcc(...)           # imported here, on first attribute access
"""

from __future__ import annotations

import importlib
from typing import Any


class LazyModule:
    """Stand-in for a module that imports it on first attribute access."""

    def __init__(self, name: str) -> None:
        self._lazy_name = name
        self._lazy_mod: Any = None

    def __getattr__(self, attr: str) -> Any:
        mod = self._lazy_mod
        if mod is None:
            mod = importlib.import_module(self._lazy_name)
            self._lazy_mod = mod
        return getattr(mod, attr)
