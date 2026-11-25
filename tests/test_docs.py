"""Guard the documentation pages against API drift.

Same approach as ``test_readme.py``: every ``from splytters... import ...`` in a
docs page must resolve, and the self-contained guide snippets must run. The
end-to-end examples in ``docs/examples/`` download datasets and models, so they
are not executed here.
"""

import re
from pathlib import Path

import numpy as np
import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"
PAGES = sorted(DOCS.glob("*.md")) + sorted((DOCS / "examples").glob("*.py"))


def _code(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".py":
        return text
    return "\n".join(re.findall(r"```python\n(.*?)```", text, re.DOTALL))


@pytest.mark.parametrize("path", PAGES, ids=lambda p: p.name)
def test_docs_imports_resolve(path):
    """Every splytters import statement in a docs page or example must resolve."""
    stmts = re.findall(
        r"^from splytters[\w.]* import (?:\([^)]*\)|[^\n]+)", _code(path), re.MULTILINE
    )
    for stmt in stmts:
        # Raises ImportError if an imported name no longer exists.
        exec(stmt, {})


def test_guide_per_class_snippet_runs():
    """The per_class_split snippet in the guide keeps each class's share."""
    from splytters import distance_adversarial_split, per_class_split

    rng = np.random.RandomState(0)
    X = rng.rand(120, 8)
    y = np.repeat([0, 1, 2], 40)
    train_idx, test_idx = per_class_split(distance_adversarial_split, X, y, train_size=0.7)
    assert np.bincount(y[train_idx]).tolist() == [28, 28, 28]
    assert len(train_idx) + len(test_idx) == len(y)


def test_guide_regression_snippet_runs():
    """The target-sorted regression split trains on the lowest targets."""
    from splytters import sorted_stratified_split

    y_reg = np.random.RandomState(0).rand(50)
    train_idx, test_idx = sorted_stratified_split(
        np.argsort(y_reg, kind="stable"), np.zeros(len(y_reg), dtype=int), train_size=0.7
    )
    assert len(train_idx) == 35
    assert y_reg[train_idx].max() < y_reg[test_idx].min()


def test_home_snippets_run():
    """The Home page snippets run in order, each using the names defined before it."""
    snippets = re.findall(r"```python\n(.*?)```", (DOCS / "index.md").read_text(encoding="utf-8"),
                          re.DOTALL)
    assert snippets
    namespace: dict = {}
    for snippet in snippets:
        exec(snippet, namespace)
