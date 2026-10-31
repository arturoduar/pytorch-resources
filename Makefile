.PHONY: install test test-fast docs docs-serve lab run serve docker-build docker-run clean

install:
	pip install -e ".[dev,notebooks,docs]"

test:
	pytest -v

# Skip the notebook execution tests.
test-fast:
	pytest -v -m "not notebooks"

docs:
	mkdocs build --strict

docs-serve:
	mkdocs serve

lab:
	jupyter lab notebooks

# Example: make run EXAMPLE=linear_regression
EXAMPLE ?= linear_regression
run:
	pytorch-examples run $(EXAMPLE) --verbose

serve:
	pytorch-examples serve --checkpoint model.pt

docker-build:
	docker build -t pytorch-examples .

docker-run:
	docker run --rm -p 8000:8000 pytorch-examples

clean:
	rm -rf site build dist .pytest_cache model.pt
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
