"""Guard the README against API drift.

Two lightweight checks (no markdown-doc plugin, no exec of every block):

1. Import parity — every ``from splytters... import ...`` shown in the README
   actually resolves, so a renamed/removed public symbol can't silently linger
   in the docs.
2. Runnable examples — the self-contained snippets (Quickstart, split_report /
   compare_splitters, a sorter) genuinely run, so the documented *signatures*
   stay correct. The illustrative blocks (model downloads, ``[...]``
   placeholders) are intentionally not executed.
"""

import re
from pathlib import Path

import numpy as np

README = Path(__file__).resolve().parent.parent / "README.md"


def _python_blocks(text: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", text, re.DOTALL)


def test_readme_imports_resolve():
    """Every splytters import statement in the README must resolve."""
    code = "\n".join(_python_blocks(README.read_text(encoding="utf-8")))
    # Single-line `from splytters[.x] import a, b` and the multi-line
    # parenthesized form `from splytters import (\n a,\n b,\n)`.
    stmts = re.findall(
        r"^from splytters[\w.]* import (?:\([^)]*\)|[^\n]+)", code, re.MULTILINE
    )
    assert stmts, "expected splytters imports in the README"
    for stmt in stmts:
        # Raises ImportError if any imported name no longer exists.
        exec(stmt, {})


def test_readme_quickstart_runs():
    """The Quickstart example runs and returns a valid split."""
    from splytters import cluster_split

    embeddings = np.random.rand(500, 384)
    train_idx, test_idx = cluster_split(embeddings, train_size=0.7)
    assert len(train_idx) + len(test_idx) == 500
    assert set(train_idx.tolist()) & set(test_idx.tolist()) == set()


def test_readme_report_and_sorter_examples_run():
    """The split_report / compare_splitters / sorter examples run as shown."""
    from splytters import compare_splitters, random_split, split_report
    from splytters.adversarial import cluster_split
    from splytters.sorters import distance_to_mean

    embeddings = np.random.RandomState(0).rand(120, 16)

    table = compare_splitters(
        embeddings, {"random": random_split, "adversarial": cluster_split}
    )
    assert {"random", "adversarial"} <= set(table)
    assert "mmd_rbf" in table["adversarial"]  # the field shown in the README

    train, test = random_split(embeddings)
    assert "coverage" in split_report(embeddings, train, test)

    ranked = distance_to_mean(embeddings)
    assert len(ranked) == len(embeddings)
