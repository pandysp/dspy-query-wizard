# Install dependencies
install:
    uv sync

# Run the application
run:
    uv run uvicorn backend.app:app --reload

# Run tests
test:
    uv run pytest

# Run linter and formatter
lint:
    uv run ruff check .
    uv run ruff format --check .
    uv run basedpyright .

# Fix linting errors
fix:
    uv run ruff check --fix .
    uv run ruff format .

# Clean up environment and cache
clean:
    rm -rf .venv
    find . -name "__pycache__" -type d -exec rm -rf {} +
