"""Regression tests: modality sorters must not *require* their heavy deps to import.

The heavy/optional libraries (torch, transformers, readability, pysbd, wordfreq,
Pillow, librosa, pandas) are imported lazily — inside the functions that use
them, or via the LazyModule proxy — so a sorter module imports (and the
dependency-free sorters run) even when the extra isn't installed.

Each check blocks the dep(s) in a fresh interpreter (a ``sys.meta_path`` finder
makes ``import dep`` raise ``ModuleNotFoundError``) and confirms the module still
imports. This is immune to transitive imports (e.g. scikit-learn pulls in pandas
when it happens to be installed).

Blocking via a meta-path finder rather than ``sys.modules[dep] = None`` matters:
the ``None`` sentinel leaves the name *present* in ``sys.modules``, so libraries
that probe ``"torch" in sys.modules`` (e.g. scipy.stats via array_api_compat at
import time) mistake it for an imported module and crash on ``getattr(None, ...)``.
The finder leaves ``sys.modules`` untouched, faithfully simulating "not installed".
"""

import subprocess
import sys

import pytest

# module -> the heavy/optional deps it must not REQUIRE in order to import.
MODALITY_DEPS = {
    "splytters.sorters.text_sorters": (
        "torch", "transformers", "readability", "pysbd", "wordfreq",
    ),
    "splytters.sorters.image_sorters": ("PIL",),
    "splytters.sorters.audio_sorters": ("librosa",),
    "splytters.sorters.tabular_sorters": ("pandas",),
}


def _run_with_deps_blocked(heavy: tuple[str, ...], body: str) -> subprocess.CompletedProcess:
    # Install a meta-path finder that raises on import of any blocked top-level
    # package, without touching sys.modules (see module docstring).
    code = (
        "import sys, importlib.abc\n"
        "class _Block(importlib.abc.MetaPathFinder):\n"
        f"    _names = set({list(heavy)!r})\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in self._names:\n"
        "            raise ModuleNotFoundError('blocked for test: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        f"{body}\n"
        "print('ok')\n"
    )
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


@pytest.mark.parametrize("module, heavy", MODALITY_DEPS.items())
def test_sorter_module_imports_with_heavy_deps_blocked(module, heavy):
    out = _run_with_deps_blocked(heavy, f"import importlib; importlib.import_module({module!r})")
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


def test_light_text_sorter_runs_with_heavy_deps_blocked():
    out = _run_with_deps_blocked(
        MODALITY_DEPS["splytters.sorters.text_sorters"],
        "from splytters.sorters import character_length; "
        "character_length(['a question', 'another one here'])",
    )
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr


def test_lazy_module_missing_dep_names_extra():
    """A missing optional dependency is re-raised naming the pip extra, instead
    of a bare ModuleNotFoundError that doesn't say how to install it."""
    from splytters.sorters._lazy import LazyModule

    proxy = LazyModule("a_module_that_does_not_exist", extra="image")
    with pytest.raises(ModuleNotFoundError, match=r"pip install splytters\[image\]"):
        _ = proxy.some_attr


def test_lazy_module_without_extra_reraises_bare():
    """Without an ``extra`` the original ModuleNotFoundError propagates unchanged."""
    from splytters.sorters._lazy import LazyModule

    proxy = LazyModule("a_module_that_does_not_exist")
    with pytest.raises(ModuleNotFoundError):
        _ = proxy.some_attr
