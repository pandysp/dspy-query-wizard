# Install dependencies
install:
    uv sync

# Run the application
run:
    uv run uvicorn backend.app:app --reload

# Run tests
test:
    uv run pytest

# Clean up environment and cache
clean:
    rm -rf .venv
    find . -name "__pycache__" -type d -exec rm -rf {} +

