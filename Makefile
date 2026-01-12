.PHONY: test lint fix

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

fix:
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/
