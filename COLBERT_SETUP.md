# ColBERT Server Setup

## Background

The public ColBERT server at `http://20.102.90.50:2017/wiki17_abstracts` has been **down since October 2024** ([GitHub Issue #8946](https://github.com/stanfordnlp/dspy/issues/8946)). It returns errors when queried.

We're using [@nielsgl's local ColBERT server](https://github.com/nielsgl/colbert-server) as a replacement, which provides the same Wikipedia 2017 index and runs locally on your machine.

## Quick Start

### Option 1: Using the startup script

```bash
./scripts/start-colbert-server.sh

# Monitor download progress
tail -f /tmp/colbert-server.log

# Stop server
pkill -f "colbert-server serve"
```

### Option 2: Manual start

```bash
# Start server (downloads 13GB on first run, then cached)
uvx colbert-server serve --from-cache --port 2017 --host 127.0.0.1

# In another terminal, test it
curl 'http://127.0.0.1:2017/api/search?query=inception&k=3'
```

## First Run: What to Expect

**⏱️ Time:** 10-30 minutes (depends on internet speed)
**💾 Download:** ~13GB Wikipedia 2017 ColBERT index from HuggingFace
**📍 Cache:** `~/.cache/huggingface/hub/`

### Download Progress

You'll see output like:
```
Fetching 848 files: 100%|██████████| 848/848 [15:23<00:00,  1.09s/it]
Loading ColBERT checkpoint...
Starting Flask server on 127.0.0.1:2017...
```

### Common Issues

**❌ Timeout errors:**
```
ReadTimeoutError: Read timed out
```
**Fix:** Retry the command. HuggingFace can be slow, it will resume from where it left off.

**❌ 401 Unauthorized:**
```
CAS service error: 401 Unauthorized
```
**Fix:** This is a temporary HuggingFace issue. Wait a few minutes and retry.

**❌ Out of disk space:**
- Needs ~15GB free space
- Delete `~/.cache/huggingface/hub/` if needed to start fresh

## Subsequent Runs

After the first successful download, the server starts **immediately** (< 5 seconds) using cached data.

## API Endpoint

```
GET http://127.0.0.1:2017/api/search?query=<text>&k=<top-k>
```

**Response format:** (to be confirmed by testing)
```json
{
  "passages": [...],
  "scores": [...],
  "probabilities": [...]
}
```

## Integration with DSPy Query Wizard

The retriever is already configured to use the local server:

**`backend/retriever.py`:**
```python
COLBERT_URL = "http://127.0.0.1:2017/api/search"
```

The app will:
1. Try ColBERT (local server)
2. Fall back to Wikipedia OpenSearch API if ColBERT is unavailable

## Alternative: Use HotPotQA Passages Directly

If the 13GB download is problematic, you can skip ColBERT entirely and use the passages included in the HotPotQA dataset:

1. Download HotPotQA (already implemented in `backend/utils/data_preprocess.py`)
2. Index the context passages in ChromaDB (~5-10 mins)
3. Use for retrieval instead

This gives you all the passages needed for multi-hop questions without the large Wikipedia download.
