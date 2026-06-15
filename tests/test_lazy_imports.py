"""Regression tests: modality sorters must not *require* their heavy deps to import.

The heavy/optional libraries (torch, transformers, readability, pysbd, wordfreq,
Pillow, librosa, pandas) are imported lazily — inside the functions that use
them, or via the LazyModule proxy — so a sorter module imports (and the
dependency-free sorters run) even when the extra isn't installed.

Each check blocks the dep(s) in a fresh interpreter (``sys.modules[dep] = None``
makes ``import dep`` raise ImportError) and confirms the module still imports.
This is immune to transitive imports (e.g. scikit-learn pulls in pandas when it
happens to be installed).
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
    blocks = "; ".join(f"sys.modules[{d!r}] = None" for d in heavy)
    code = f"import sys; {blocks}; {body}; print('ok')"
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
