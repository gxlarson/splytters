# Contributing to splytters

Thanks for your interest in improving splytters! This guide covers local setup,
tests, and conventions.

## Development setup

We recommend [`uv`](https://github.com/astral-sh/uv) (fast), but plain `venv` +
`pip` works too. Python **3.10+** is required.

```bash
# with uv
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"     # everything, incl. test deps

# or with pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

For fast iteration on the core splitting algorithms you only need the core
install — no heavy extras:

```bash
pip install -e . pytest
pytest tests/test_splitters.py tests/test_interop.py tests/test_report.py
```

### Git hooks (recommended)

The repo uses [pre-commit](https://pre-commit.com) to run two checks on every
commit (config in `.pre-commit-config.yaml`):

- **gitleaks** scans staged changes for credentials (API keys, tokens, private
  keys) and blocks the commit before the secret enters history.
- **ruff** lints the staged Python files, the same check as CI's `lint` job.

Activate it once per clone:

```bash
pip install pre-commit
pre-commit install
```

The first commit afterward takes a minute while pre-commit builds the hook
environments. If gitleaks flags a confirmed false positive, add the finding's
fingerprint (printed in the hook output) to `.gitleaksignore`. CI runs gitleaks
over the full history on every push and PR, and GitHub-side secret scanning
push protection is enabled on the repository as another layer.

If you previously activated the old `pre-push` hook, clear it first, since
`pre-commit install` refuses to run while it is set:

```bash
git config --unset core.hooksPath
```

## Running tests

```bash
pytest                 # full suite
pytest -m "not slow"   # skip model-download tests (perplexity etc.)
ruff check .           # lint
```

- A **core-only** install (`pip install -e .`) collects and runs cleanly:
  `conftest.py` skips the heavy-modality sorter test modules whose optional
  dependency (librosa, Pillow, torch, …) is absent.
- Tests must be **deterministic** — `conftest.py` seeds numpy/`random` before
  each test. Always pass a fixed `random_state` in new tests.
- Mark slow/network tests with `@pytest.mark.slow`.

## Conventions

- **Splitters** take `(embeddings, train_size=0.7, random_state=42, ...)` and
  return a `(train_indices, test_indices)` pair of integer `ndarray`s that
  partition `range(n_samples)`. Validate inputs through
  `splytters.utils.validate_split_inputs` (handles `check_array`, finite
  checks, and `train_size` as a fraction *or* absolute count).
- **Returns are ndarrays**, parameters follow scikit-learn naming
  (`train_size`, `random_state`, `y`).
- **Optional dependencies** must be imported lazily (inside the function or via
  the package `__getattr__`) so each extra is self-sufficient.
- Add a NumPy-style docstring and at least one test (valid-split invariants +
  determinism) for every new splitter.

## Reproducible environments

CI's `full` job runs against the *latest* release of every optional dependency
(a canary, allowed to fail on upstream API drift). For reproducing the
experiment numbers, pin versions with a constraints file:

```bash
pip install -e ".[dev]" -c constraints.txt
```

## Pull requests

1. Branch off `main` (e.g. `feature/my-splitter`).
2. Keep the core (`pip install -e .`) test job green.
3. Update `CHANGELOG.md` under "Unreleased".
4. Run `ruff check .` and `pytest -m "not slow"` before pushing (the
   pre-commit hook covers ruff for staged files).
