.PHONY: test lint fix proto

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

fix:
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

proto:
	uv run python -m grpc_tools.protoc \
		-I proto \
		--python_out=src/privateboost/grpc \
		--pyi_out=src/privateboost/grpc \
		--grpc_python_out=src/privateboost/grpc \
		proto/privateboost.proto
