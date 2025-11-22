# Quick Start Guide

## Setup (One-time)

```bash
# 1. Install Python dependencies
just install

# 2. Start ColBERT server (downloads 13GB Wikipedia index on first run)
just colbert-start

# Monitor the download progress in another terminal:
just colbert-logs

# When you see "Starting Flask server..." the download is complete
```

**First run:** 10-30 minutes for the 13GB download
**Subsequent runs:** Server starts immediately (< 5 seconds)

## Configuration

1.  **Copy the example environment file:**
    ```bash
    cp .env.example .env
    ```
2.  **Add your API Keys:**
    Open `.env` and set your `OPENAI_API_KEY`. This is required for:
    *   Running the DSPy optimizer (Training).
    *   Using the Machine RAG pipeline with a real LLM.

## Daily Workflow

```bash
# Terminal 1: Start ColBERT server (if not already running)
just colbert-start

# Terminal 2: Start FastAPI backend
just run

# Terminal 3: Run frontend (optional)
cd frontend && pnpm dev
```

## Training (Optional)

To optimize the Machine RAG pipeline using HotPotQA data:

1.  **Download Dataset:**
    ```bash
    uv run python backend/utils/data_preprocess.py
    ```
2.  **Run Training:**
    ```bash
    uv run python backend/train.py
    ```
    *Requires `OPENAI_API_KEY` in `.env`.*

## Common Commands

```bash
# Check if ColBERT server is responding
just colbert-status

# Test retrieval manually
curl 'http://127.0.0.1:2017/api/search?query=Christopher+Nolan&k=3'

# Test the FastAPI endpoint
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Who is the director of Inception?"}'

# Run tests
just test

# Stop ColBERT server
just colbert-stop
```

## Troubleshooting

**ColBERT server won't start:**
- Check logs: `just colbert-logs`
- Kill existing processes: `just colbert-stop`
- Try again: `just colbert-start`

**Download timeouts:**
- Normal! HuggingFace can be slow
- The download will resume from where it left off
- Just restart: `just colbert-start`

**Out of disk space:**
- Need ~15GB free
- Cache location: `~/.cache/huggingface/hub/`

## File Structure

```
backend/
  app.py              # FastAPI server
  retriever.py        # ColBERT + Wikipedia retrieval logic
  utils/
    data_preprocess.py # HotPotQA dataset download

scripts/
  start-colbert-server.sh # ColBERT server startup script

tests/
  test_retriever.py   # Retrieval tests

COLBERT_SETUP.md      # Detailed ColBERT setup guide
AGENTS.md             # Full project documentation
```

## Next Steps

1. ✅ ColBERT server running
2. ⏳ Update retriever to handle nielsgl/colbert-server response format
3. ⏳ Test end-to-end retrieval
4. ⏳ Implement DSPy optimization pipeline
5. ⏳ Build frontend comparison UI
