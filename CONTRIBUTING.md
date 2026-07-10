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

The repo ships a `pre-push` hook that scans outgoing commits for credential
patterns (AWS/GitHub/OpenAI/Anthropic/Google/Slack keys, private key blocks,
generic secret assignments) and blocks the push before anything leaves your
machine. It is plain POSIX `sh` with no extra dependencies and works in Git
Bash on Windows. Activate it once per clone:

```bash
git config core.hooksPath scripts/hooks
```

If it flags a confirmed false positive, bypass a single push with
`git push --no-verify`. GitHub-side secret scanning push protection is also
enabled on the repository as a second layer.

## Running tests

```bash
pytest                 # full suite
pytest -m "not slow"   # skip model-download tests (perplexity etc.)
ruff check splytters   # lint
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
4. Run `ruff check splytters` and `pytest -m "not slow"` before pushing.
