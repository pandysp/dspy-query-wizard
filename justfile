# Install dependencies
install:
    uv sync

# Start ColBERT server (required for retrieval)
colbert-start:
    ./scripts/start-colbert-server.sh

# Stop ColBERT server
colbert-stop:
    pkill -f "colbert-server serve" || echo "No ColBERT server running"

# Check ColBERT server status
colbert-status:
    @curl -s 'http://127.0.0.1:2017/api/search?query=test&k=1' > /dev/null && echo "✅ ColBERT server is running" || echo "❌ ColBERT server is not responding"

# View ColBERT server logs
colbert-logs:
    tail -f /tmp/colbert-server.log

# Test ColBERT server end-to-end
colbert-test:
    ./scripts/test-colbert.sh

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
