"""Lazy module proxy for the modality sorters.

Some modality sorters reference a heavy, optional dependency (librosa, Pillow,
pandas) pervasively across their functions. Importing it at module load would
make e.g. ``import splytters.sorters.audio_sorters`` require librosa even just
to read a docstring or list the sorters. Binding the dependency to a
``LazyModule`` proxy defers the import to first use, so importing a sorter
module stays dependency-free; only *calling* a sorter pulls the library in.

Usage::

    librosa = LazyModule("librosa", extra="audio")   # instead of: import librosa
    ...
    librosa.feature.mfcc(...)           # imported here, on first attribute access

Passing ``extra`` turns a missing dependency into an actionable error naming the
pip extra to install (e.g. ``pip install splytters[audio]``) instead of a bare
``ModuleNotFoundError: No module named 'librosa'``.
"""

from __future__ import annotations

import importlib
from typing import Any


class LazyModule:
    """Stand-in for a module that imports it on first attribute access.

    Args:
        name: the importable module name (e.g. ``"librosa"``, ``"PIL.Image"``).
        extra: the splytters optional-dependency extra that provides this module
            (e.g. ``"audio"``). When set, a missing dependency is re-raised with
            a message naming the extra to install.
    """

    def __init__(self, name: str, extra: str | None = None) -> None:
        self._lazy_name = name
        self._lazy_extra = extra
        self._lazy_mod: Any = None

    def __getattr__(self, attr: str) -> Any:
        mod = self._lazy_mod
        if mod is None:
            try:
                mod = importlib.import_module(self._lazy_name)
            except ModuleNotFoundError as exc:
                if self._lazy_extra is None:
                    raise
                top = self._lazy_name.split(".")[0]
                raise ModuleNotFoundError(
                    f"{top} is required for {self._lazy_extra} sorters -- install "
                    f"with: pip install splytters[{self._lazy_extra}]"
                ) from exc
            self._lazy_mod = mod
        return getattr(mod, attr)
