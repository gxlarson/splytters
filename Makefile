.PHONY: install test test-core lint clean

install:
	pip install -e ".[dev]"

# Full suite minus slow model-download tests.
test:
	pytest -m "not slow"

# Core-only contract: splitters + interop + report (no heavy extras needed).
test-core:
	pytest tests/test_splitters.py tests/test_interop.py tests/test_report.py

lint:
	ruff check .

clean:
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
