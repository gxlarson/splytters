.PHONY: install test test-core lint figures validate clean

install:
	pip install -e ".[dev]"

# Full suite minus slow model-download tests.
test:
	pytest -m "not slow"

# Core-only contract: splitters + interop + report (no heavy extras needed).
test-core:
	pytest tests/test_splitters.py tests/test_interop.py tests/test_report.py

lint:
	ruff check splytters

# Regenerate the paper's figures + CSVs (offline / cached embeddings, fixed seeds).
figures:
	python experiments/run_experiment.py --dataset synth  --seeds 10 --out experiments/results_synth
	python experiments/run_experiment.py --dataset digits --seeds 10 --out experiments/results

# Full multi-dataset (incl. real text) × multi-model validation + the
# energy-distance↔difficulty correlation. Needs the [demo] extra for newsgroups
# (cached after first run); pass --datasets synth digits to stay fully offline.
validate:
	python experiments/validate.py --seeds 5 --out experiments/results

clean:
	rm -rf experiments/results .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
