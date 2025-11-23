set shell := ["powershell.exe", "-NoLogo", "-ExecutionPolicy", "Bypass", "-Command"]
# or if you're using PowerShell Core:
# set shell := ["pwsh", "-NoLogo", "-ExecutionPolicy", "Bypass", "-Command"]


# Install dependencies
install:
    uv sync

# Start ColBERT server (required for retrieval)
colbert-start:
    ./scripts/start-colbert-server.ps1

# Stop ColBERT server
colbert-stop:
    Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*backend/colbert_server.py*" } | Stop-Process -Force
    # Also kill the cmd wrapper if possible, but python is the main one.
    # Or just tell user to kill it.
    Write-Host "Attempted to stop server. Verify with colbert-status."

# Check ColBERT server status
colbert-status:
    @try { Invoke-RestMethod 'http://127.0.0.1:2017/api/search?query=test&k=1' -ErrorAction Stop | Out-Null; Write-Host "✅ ColBERT server is running" } catch { Write-Host "❌ ColBERT server is not responding" }

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
