.PHONY: docs html lint format test install

install:
	pip install -e ".[dev,docs]"

test:
	pytest tests -v --tb=short

lint:
	black --check src tests
	isort --check-only src tests
	flake8 src tests

format:
	black src tests
	isort src tests

html:
	cd docs && sphinx-build -b html . _build/html

docs: html
