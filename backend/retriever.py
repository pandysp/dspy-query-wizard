import httpx
import dspy  # type: ignore
from dspy.dsp.utils import dotdict  # type: ignore
from typing import Any, Optional, Union, override
from joblib import Memory  # type: ignore
import os
import asyncio

# --- Configuration ---
COLBERT_URL = "http://20.102.90.50:2017/wiki17_abstracts"
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


@memory.cache
def _cached_retrieval_sync(query: str, k: int) -> list[dict[str, Any]]:
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
            data = resp.json()
            if isinstance(data, dict) and data.get("error") is True:
                raise Exception("Server Error")

            if "topk" in data:
                return data["topk"][:k]
            elif "passages" in data:
                # RAGatouille/other format normalization
                passages = data["passages"]
                scores = data.get("scores", [])
                pids = data.get("pids", [])
                return [
                    {
                        "text": p,
                        "pid": pids[i] if i < len(pids) else None,
                        "score": scores[i] if i < len(scores) else None,
                    }
                    for i, p in enumerate(passages[:k])
                ]
        except Exception:
            pass  # Proceed to fallback

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
            wiki_data = wiki_resp.json()
            if not wiki_data or len(wiki_data) < 4:
                return []

            titles, descriptions, urls = wiki_data[1], wiki_data[2], wiki_data[3]
            results = []
            for i, title in enumerate(titles):
                text = (
                    f"Title: {title}\nSummary: {descriptions[i]}"
                    if descriptions[i]
                    else f"Title: {title}"
                )
                results.append(
                    {
                        "text": text,
                        "pid": f"wiki-{title}",
                        "score": 1.0 - (i * 0.1),
                        "url": urls[i],
                    }
                )
            return results
        except Exception as e:
            print(f"[Retriever] All methods failed for '{query}': {e}")
            return [{"text": "Failed to retrieve", "pid": -1, "score": 0.0}]


async def fetch_colbert_results(
    client: Optional[httpx.AsyncClient], query: str, k: int = 5
) -> list[dict[str, Any]]:
    """
    Primary Retriever Entrypoint (Async).
    Now uses the cached sync function wrapped in asyncio.to_thread to handle I/O + Caching.
    """
    # We offload the cached sync call to a thread because joblib is blocking
    # This keeps our FastAPI event loop unblocked while getting the benefit of file-based caching
    return await asyncio.to_thread(_cached_retrieval_sync, query, k)


async def prewarm_cache():
    """
    Fires off requests for known demo questions to populate the cache.
    """
    print("[Cache] Pre-warming started...")
    # Use a throwaway async client just for consistency, but we rely on the sync cached function via thread
    # Actually, fetch_colbert_results handles the client arg, but ignores it for the actual logic
    # since it calls _cached_retrieval_sync which makes its own sync client.
    # We can just call fetch_colbert_results with None as client since we refactored it to use to_thread.

    tasks = []
    for q in PREWARM_QUESTIONS:
        tasks.append(
            fetch_colbert_results(None, q, k=3)
        )  # Client is ignored in the new impl

    await asyncio.gather(*tasks)
    print("[Cache] Pre-warming complete.")


class AsyncColBERTv2RM(dspy.Retrieve):
    """
    DSPy-compatible wrapper for the optimizer loop (Synchronous).
    """

    def __init__(self, k: int = 3):
        super().__init__(k=k)

    # Updated signature to match base class
    @override
    def forward(
        self,
        query: Union[str, list[str]],
        k: Optional[int] = None,
        **kwargs: Any,
    ) -> dspy.Prediction:
        k = k if k is not None else self.k
        # Handle single string vs list of strings
        queries = [query] if isinstance(query, str) else query

        passages = []
        for q in queries:
            results = _cached_retrieval_sync(q, k)
            for r in results:
                passages.append(
                    dotdict(
                        {
                            "long_text": r.get("text", ""),
                            "pid": r.get("pid"),
                            "score": r.get("score", 0),
                        }
                    )
                )

        # Pyright complains that list[dotdict] isn't dspy.Prediction.
        # In DSPy, forward typically returns a Prediction object containing 'passages'
        # OR a list of strings/Prediction objects depending on usage.
        # The base class signature says -> Prediction.
        # We will return it as `dspy.Prediction` (which is just a wrapper) to satisfy the type checker.
        return dspy.Prediction(passages=passages)


# Configure Global Retriever for DSPy
rm = AsyncColBERTv2RM()
dspy.settings.configure(rm=rm)
