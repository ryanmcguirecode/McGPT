init:
    if ! command -v uv > /dev/null; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh; \
    fi
    uv sync