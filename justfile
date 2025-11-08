init:
    if ! command -v uv > /dev/null; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh; \
    fi
    uv sync

add package:
    uv add {{package}} --index-strategy unsafe-best-match

build:
    just format
    just lint-fix
    just test

test:
    uv run bash -c "PYTHONPATH=src pytest -q"

format:
    uv run black src tests

check-format:
    uv run black --check src tests

lint:
    uv run ruff check --select F,I --fix --unsafe-fixes src tests

lint-fix:
    uv run -- ruff check --fix src tests