# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DSPy Query Wizard** is a comparative RAG (Retrieval-Augmented Generation) system demonstrating the difference between manual prompt engineering ("Human") and automated prompt optimization via DSPy ("Machine"). The system retrieves documents from ColBERTv2/Wikipedia and uses the HotPotQA dataset for multi-hop question answering.

**Key Technologies:**
- **Backend:** FastAPI + DSPy + httpx (async I/O)
- **Frontend:** React + TypeScript + Vite + Tailwind CSS
- **Package Manager:** `uv` (Python), `pnpm` (frontend)
- **Task Runner:** `just` (justfile)
- **Ticketing:** `bd` (Beads ticketing system)

## Development Commands

### Backend (Python)

```bash
# Install dependencies
just install

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

## Architecture

### Backend Structure

The backend uses a **functional-first approach** with async/await throughout:

**`backend/app.py`** - FastAPI application with lifespan management:
- Prewarms cache on startup with "evil questions" (known complex queries)
- Single endpoint `/api/query` that accepts questions and returns retrieved contexts
- Currently returns placeholder responses for human/machine answers

**`backend/retriever.py`** - Core retrieval logic with resilient fallback:
- **Primary:** ColBERTv2 server (`http://20.102.90.50:2017/wiki17_abstracts`)
- **Fallback:** Wikipedia OpenSearch API
- **Caching:** Uses `joblib.Memory` for file-based caching (`.cache/` directory)
- **`fetch_colbert_results()`** - Async entrypoint wrapping cached sync retrieval
- **`AsyncColBERTv2RM`** - DSPy-compatible retriever class for optimizer loop (synchronous)
- **`prewarm_cache()`** - Preloads cache with demo questions on startup

**`backend/utils/data_preprocess.py`** - HotPotQA dataset download utility:
- Downloads HotPotQA fullwiki dataset
- Saves train/eval/test splits to `backend/data/`

### Key Design Patterns

1. **Async/Sync Hybrid for Caching:**
   - `_cached_retrieval_sync()` is the joblib-cached function (sync, uses httpx.Client)
   - `fetch_colbert_results()` wraps it with `asyncio.to_thread()` for FastAPI
   - Both code paths share the same file cache

2. **Resilient Retrieval:**
   - Always try ColBERT first
   - Silent fallback to Wikipedia on any ColBERT failure
   - Returns fallback message if both fail
   - No crashes, always returns a result

3. **Colocation by Feature:**
   - Retrieval logic (ColBERT + Wikipedia + caching) lives in one file
   - Avoid splitting related logic into separate utils/helpers

### Frontend Structure

React 19 + TypeScript + Vite with Tailwind CSS 4. Currently minimal scaffolding.

### Testing

Tests use `pytest` with `pytest-asyncio` for async support. Tests mock httpx.Client to avoid external dependencies.

## Coding Guidelines (from AGENTS.md)

- **Functional > OOP:** Use classes only when frameworks (DSPy) require it
- **Async/await + httpx:** Never use `requests`, always use `httpx` with async
- **Resilience:** Explicit retries, fallbacks, error handling for external services
- **Colocation:** Group code by feature/domain, not technical layer
- **Task Tracking:** Use `bd` for all planning/tickets, NOT internal todo tools
- **Verification:** Test immediately after implementation (`curl`, `pytest`, `just test`)
- **Boy Scout Rule:** Create tickets for tech debt/bugs found while working

## Configuration

- **Python:** 3.12+ (see `.python-version`)
- **Type Checking:** basedpyright (more lenient than pyright)
- **Linting:** ruff
- **Dependencies:** Managed by `uv` (see `pyproject.toml` and `uv.lock`)

## Important Notes

- **Cache warming:** The app preloads 5 "evil questions" on startup to ensure demo responsiveness
- **External dependency:** ColBERT server may be down; Wikipedia fallback ensures the app always works
- **DSPy integration:** The `AsyncColBERTv2RM` class is configured globally via `dspy.settings.configure(rm=rm)`
- **No venv/pip:** Always use `uv` commands (`uv run`, `uv sync`)
