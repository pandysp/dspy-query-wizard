import httpx
import dspy
from dspy.dsp.utils import dotdict
from typing import List, Dict, Any, Union
from joblib import Memory
import os

# --- Configuration ---
COLBERT_URL = "http://20.102.90.50:2017/wiki17_abstracts"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "DSPy-Query-Wizard/1.0 (https://github.com/yourusername/dspy-query-wizard; contact@example.com)"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "../.cache")

# --- Caching Setup ---
memory = Memory(CACHE_DIR, verbose=0)

@memory.cache
def _cached_retrieval_sync(query: str, k: int) -> List[Dict[str, Any]]:
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
                          "score": scores[i] if i < len(scores) else None
                      } 
                      for i, p in enumerate(passages[:k])
                  ]
        except Exception:
            pass # Proceed to fallback
            
        try:
            # 2. Wikipedia Fallback
            headers = {"User-Agent": USER_AGENT}
            wiki_resp = client.get(WIKIPEDIA_API_URL, params={"action": "opensearch", "search": query, "limit": k, "namespace": 0, "format": "json"}, headers=headers, timeout=3.0)
            wiki_data = wiki_resp.json()
            if not wiki_data or len(wiki_data) < 4:
                return []
            
            titles, descriptions, urls = wiki_data[1], wiki_data[2], wiki_data[3]
            results = []
            for i, title in enumerate(titles):
                text = f"Title: {title}\nSummary: {descriptions[i]}" if descriptions[i] else f"Title: {title}"
                results.append({
                    "text": text,
                    "pid": f"wiki-{title}",
                    "score": 1.0 - (i * 0.1),
                    "url": urls[i]
                })
            return results
        except Exception as e:
             print(f"[Retriever] All methods failed for '{query}': {e}")
             return [{"text": "Failed to retrieve", "pid": -1, "score": 0.0}]


async def fetch_colbert_results(client: httpx.AsyncClient, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Primary Retriever Entrypoint (Async).
    Now uses the cached sync function wrapped in asyncio.to_thread to handle I/O + Caching.
    """
    import asyncio
    # We offload the cached sync call to a thread because joblib is blocking
    # This keeps our FastAPI event loop unblocked while getting the benefit of file-based caching
    return await asyncio.to_thread(_cached_retrieval_sync, query, k)


class AsyncColBERTv2RM(dspy.Retrieve):
    """
    DSPy-compatible wrapper for the optimizer loop (Synchronous).
    """
    def __init__(self, k: int = 3):
        super().__init__(k=k)
    
    def forward(self, query_or_queries: Union[str, List[str]], k: int | None = None) -> dspy.Prediction:
        k = k if k is not None else self.k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        
        passages = []
        for q in queries:
             results = _cached_retrieval_sync(q, k)
             for r in results:
                 passages.append(dotdict({"long_text": r.get("text", ""), "pid": r.get("pid"), "score": r.get("score", 0)}))
        
        return passages

# Configure Global Retriever for DSPy
rm = AsyncColBERTv2RM()
dspy.settings.configure(rm=rm)

