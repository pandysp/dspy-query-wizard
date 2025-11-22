# DSPy Query Wizard

## Project Overview

**DSPy Query Wizard** is a comparative RAG (Retrieval-Augmented Generation) system demonstrating the difference between manual prompt engineering ("Human") and automated prompt optimization via DSPy ("Machine"). The system retrieves documents from ColBERTv2/Wikipedia and uses the HotPotQA dataset for multi-hop question answering.

**Key Technologies:**
- **Backend:** FastAPI + DSPy + httpx (async I/O)
- **Frontend:** React + TypeScript + Vite + Tailwind CSS

---

## User Preferences & Workflow

### Tooling & Configuration
- **Language:** Python 3.12+
- **Package Manager:** `uv` (avoid `pip` / `venv` unless necessary).
- **Task Runner:** `just` (use `justfile`, never `Makefile`).
- **Ticketing:** `bd` (Epics > Features > Tasks).
- **Type Checking:** `basedpyright` (more lenient than pyright).
- **Linting:** `ruff`.

### Coding Style
- **Paradigm:** Functional > OOP. Use classes only when frameworks (e.g., DSPy) strictly enforce it.
- **Async/IO:** Always use `async/await` and `httpx`. Avoid `requests`.
- **Resilience:** Implement explicit retries, fallbacks, and robust error handling for external services.
- **Colocation:** "What changes together belongs together." Group code by feature/domain, not by technical layer. Avoid scattering related logic across multiple "utils" or "helpers" files.
- **WTFs/Minute:** Minimize "WTFs/Minute". Write clean, standard, unsurprising code. Refactor hacks immediately.

### Workflow
- **Red, Green, Refactor (RGR):** Unless specifically instructed otherwise, always follow the RGR cycle: 1. Write a failing test (Red). 2. Write the minimum code to make the test pass (Green). 3. Improve the code without changing its behavior (Refactor). Always run relevant tests after each 'Green' or 'Refactor' step.
- **Planning:** Brainstorm architecture and "happy paths" before coding. Use `bd` for all task tracking. Do NOT use internal todo tools/memory.
- **Verification:** Always test immediately (e.g., `curl`, `pytest`, `just test`) after implementation.
- **Communication:** Be concise. Focus on "Changes Made" and "Next Steps". Avoid verbose pleasantries.
- **Boy Scout Rule:** If you spot issues (tech debt, messiness, bugs) while working, create a ticket immediately. Capture improvements as you go.

---

## Architecture

### Backend Structure

The backend uses a **functional-first approach** with async/await throughout.

**`backend/app.py`** - FastAPI application with lifespan management:
- Prewarms cache on startup with "evil questions" (known complex queries) to ensure responsiveness.
- Single endpoint `/api/query` that accepts questions and returns retrieved contexts.
- Currently returns placeholder responses for human/machine answers.

**`backend/retriever.py`** - Core retrieval logic with resilient fallback:
- **Primary:** Local ColBERT server (`http://127.0.0.1:2017/api/search`) using [nielsgl/colbert-server](https://github.com/nielsgl/colbert-server).
  - Start with: `just colbert-start` (downloads ~13GB Wikipedia 2017 index on first run, then cached)
  - See `COLBERT_SETUP.md` for details
- **Fallback:** Wikipedia OpenSearch API (ensures system works even if ColBERT is down).
- **Caching:** Uses `joblib.Memory` for file-based caching (`.cache/` directory).
- **`fetch_colbert_results()`** - Async entrypoint wrapping cached sync retrieval.
- **`AsyncColBERTv2RM`** - DSPy-compatible retriever class for optimizer loop (synchronous), configured globally via `dspy.settings`.

**`backend/utils/data_preprocess.py`** - HotPotQA dataset download utility:
- Downloads HotPotQA fullwiki dataset.
- Saves train/eval/test splits to `backend/data/`.

### Key Design Patterns

1. **Async/Sync Hybrid for Caching:**
   - `_cached_retrieval_sync()` is the joblib-cached function (sync, uses httpx.Client).
   - `fetch_colbert_results()` wraps it with `asyncio.to_thread()` for FastAPI.
   - Both code paths share the same file cache.

2. **Resilient Retrieval:**
   - Always try ColBERT first.
   - Silent fallback to Wikipedia on any ColBERT failure.
   - Returns fallback message if both fail.
   - No crashes, always returns a result.

3. **Colocation by Feature:**
   - Retrieval logic (ColBERT + Wikipedia + caching) lives in one file.
   - Avoid splitting related logic into separate utils/helpers.

### Frontend Structure

React 19 + TypeScript + Vite + Tailwind CSS 4. Currently minimal scaffolding.

### Testing

Tests use `pytest` with `pytest-asyncio` for async support. Tests mock httpx.Client to avoid external dependencies.

---

## Development Commands

### Backend (Python)

```bash
# Install dependencies
just install

# Start ColBERT retrieval server (required - see COLBERT_SETUP.md)
just colbert-start       # Downloads ~13GB on first run, then cached
just colbert-status      # Check if server is running
just colbert-stop        # Stop the server
just colbert-logs        # View server logs

# Run the FastAPI server (auto-reload enabled)
just run

# Run all tests
just test

# Run a single test file
uv run pytest tests/test_retriever.py

# Run a specific test function
uv run pytest tests/test_retriever.py::test_colbert_success

# Lint and type-check
just lint

# Auto-fix linting issues
just fix

# Clean environment
just clean
```

### Frontend (React)

```bash
cd frontend

# Install dependencies
pnpm install

# Run dev server
pnpm dev

# Build for production
pnpm build

# Lint
pnpm lint

# Preview production build
pnpm preview
```
