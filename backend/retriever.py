import httpx
from typing import Any, NotRequired, TypedDict
from joblib import Memory  # type: ignore
import os
import asyncio
import logging

# Set up logging
logger = logging.getLogger(__name__)


# Type definitions for API responses
class RetrievalResult(TypedDict):
    """Result from retrieval (ColBERT or Wikipedia)."""

    text: str
    pid: str | int
    score: float
    url: NotRequired[str]  # Only present in Wikipedia results


# --- Configuration ---
# Local ColBERT server (https://github.com/nielsgl/colbert-server)
# Start with: uvx colbert-server serve --from-cache --port 2017
COLBERT_URL = os.getenv("COLBERT_URL", "http://127.0.0.1:2017/api/search")
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "DSPy-Query-Wizard/1.0 (https://github.com/yourusername/dspy-query-wizard; contact@example.com)"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "../.cache")

# "Evil Questions" to pre-warm the cache with
PREWARM_QUESTIONS = [
    "Who is the director of Inception?",
    "When was the father of the director of Inception born?",
    "What awards has Leonardo DiCaprio won?",
    "Christopher Nolan filmography",
    "Barack Obama citizenship conspiracy theories",
]

# --- Caching Setup ---
memory = Memory(CACHE_DIR, verbose=0)


@memory.cache  # type: ignore[misc]
def _cached_retrieval_sync(query: str, k: int) -> list[RetrievalResult]:
    """
    Synchronous cached retrieval function.
    Used by both the async wrapper (via thread) and the sync wrapper.
    This ensures we share the cache between both modes.
    """
    # Note: joblib cache works on function arguments.
    # We use a sync httpx client here because joblib is sync-friendly.
    with httpx.Client() as client:
        try:
            # 1. ColBERT
            resp = client.get(COLBERT_URL, params={"query": query, "k": k}, timeout=2.0)
            # resp.json() returns Any from untyped httpx, but we know it's dict-like
            data: Any = resp.json()
            if data.get("error") is True:
                raise Exception("Server Error")

            if "topk" in data:
                # ColBERT returns list of passages with text/pid/score
                return data["topk"][:k]
            elif "passages" in data:
                # RAGatouille/other format normalization
                passages: list[str] = data["passages"]
                scores: list[float] = data.get("scores", [])
                pids: list[Any] = data.get("pids", [])
                return [
                    RetrievalResult(
                        text=p,
                        pid=pids[i] if i < len(pids) else -1,
                        score=scores[i] if i < len(scores) else 0.0,
                    )
                    for i, p in enumerate(passages[:k])
                ]
        except Exception as e:
            logger.warning(f"ColBERT retrieval failed for '{query}': {e}")
            # Proceed to fallback

        try:
            # 2. Wikipedia Fallback
            headers = {"User-Agent": USER_AGENT}
            wiki_resp = client.get(
                WIKIPEDIA_API_URL,
                params={
                    "action": "opensearch",
                    "search": query,
                    "limit": k,
                    "namespace": 0,
                    "format": "json",
                },
                headers=headers,
                timeout=3.0,
            )
            # Wikipedia OpenSearch returns [query, [titles], [descriptions], [urls]]
            wiki_data: Any = wiki_resp.json()
            if not wiki_data or len(wiki_data) < 4:
                return []

            titles: list[str] = wiki_data[1]
            descriptions: list[str] = wiki_data[2]
            urls: list[str] = wiki_data[3]
            results: list[RetrievalResult] = []
            for i, title in enumerate(titles):
                text = (
                    f"Title: {title}\nSummary: {descriptions[i]}"
                    if descriptions[i]
                    else f"Title: {title}"
                )
                results.append(
                    RetrievalResult(
                        text=text,
                        pid=f"wiki-{title}",
                        score=1.0 - (i * 0.1),
                        url=urls[i],
                    )
                )
            return results
        except Exception as e:
            logger.error(f"All retrieval methods failed for '{query}': {e}")
            return [RetrievalResult(text="Failed to retrieve", pid=-1, score=0.0)]


def search_wikipedia(query: str, k: int = 3) -> list[str]:
    """
    Search Wikipedia for documents relevant to the query.
    Use this tool to find facts, dates, places, and people.
    Returns a list of relevant text passages.
    """
    results = _cached_retrieval_sync(query, k)
    return [r["text"] for r in results]


async def fetch_colbert_results(query: str, k: int = 5) -> list[RetrievalResult]:
    """
    Primary Retriever Entrypoint (Async).
    Now uses the cached sync function wrapped in asyncio.to_thread to handle I/O + Caching.
    """
    # We offload the cached sync call to a thread because joblib is blocking
    # This keeps our FastAPI event loop unblocked while getting the benefit of file-based caching
    return await asyncio.to_thread(_cached_retrieval_sync, query, k)


async def prewarm_cache() -> None:
    """
    Fires off requests for known demo questions to populate the cache.
    """
    logger.info("Cache Pre-warming started...")

    tasks = [fetch_colbert_results(q, k=3) for q in PREWARM_QUESTIONS]
    _ = await asyncio.gather(*tasks)
    logger.info("Cache Pre-warming complete.")