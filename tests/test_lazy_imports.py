"""Regression tests: modality sorters must not eagerly import heavy deps.

Importing a sorter module (or a lightweight sorter) must work even when the
heavy/optional libraries are missing or broken — they are imported lazily
inside the functions that need them. Run in a fresh interpreter so the check
is unaffected by whatever the rest of the suite has already imported.
"""

import subprocess
import sys

HEAVY = ("torch", "transformers", "readability", "pysbd", "wordfreq")


def _heavy_modules_after(import_stmt: str) -> str:
    code = (
        f"import sys; {import_stmt}; "
        f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_text_sorters_module_imports_without_heavy_deps():
    assert _heavy_modules_after("import splytters.sorters.text_sorters") == ""


def test_light_text_sorter_runs_without_heavy_deps():
    stmt = (
        "from splytters.sorters import character_length; "
        "character_length(['a question', 'another one here'])"
    )
    assert _heavy_modules_after(stmt) == ""
