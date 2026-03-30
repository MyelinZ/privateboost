.PHONY: test lint fix figures

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

fix:
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

figures:
	cd notebooks && uv run jupyter nbconvert --to notebook --execute generate_figures.ipynb --output generate_figures.ipynb
