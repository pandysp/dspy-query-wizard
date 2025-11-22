import pytest
from unittest.mock import patch, MagicMock
from backend.retriever import fetch_colbert_results, COLBERT_URL, WIKIPEDIA_API_URL
from typing import Any, cast
import time


@pytest.mark.asyncio
async def test_colbert_success():
    # Use timestamp to ensure truly unique query that won't hit cache
    query = f"unique_query_colbert_success_{time.time()}"

    with patch("backend.retriever.httpx.Client") as mock_client_cls:
        mock_client = cast(MagicMock, mock_client_cls.return_value)
        mock_client.__enter__.return_value = mock_client  # type: ignore

        # Mock ColBERT response
        mock_resp = MagicMock()
        mock_resp.json.return_value = {  # type: ignore
            "topk": [
                {"text": "Passage 1", "pid": 1, "score": 10.0},
                {"text": "Passage 2", "pid": 2, "score": 9.0},
            ]
        }
        mock_client.get.return_value = mock_resp  # type: ignore

        results = await fetch_colbert_results(query, k=2)

        assert len(results) == 2
        assert results[0]["text"] == "Passage 1"

        # Verify ColBERT URL was called with correct params
        # Note: assert_called_with might be tricky if get is called multiple times or differently
        # checking strictly for the call arguments
        call_args = mock_client.get.call_args  # type: ignore
        assert call_args[0][0] == COLBERT_URL  # type: ignore
        assert call_args[1]["params"]["query"] == query  # type: ignore
        assert call_args[1]["params"]["k"] == 2  # type: ignore


@pytest.mark.asyncio
async def test_fallback_to_wikipedia():
    # Use timestamp to ensure truly unique query that won't hit cache
    query = f"unique_query_fallback_{time.time()}"

    with patch("backend.retriever.httpx.Client") as mock_client_cls:
        mock_client = cast(MagicMock, mock_client_cls.return_value)
        mock_client.__enter__.return_value = mock_client  # type: ignore

        # Mock ColBERT failure then Wikipedia success
        def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            _ = kwargs
            url = cast(str, args[0])
            if url == COLBERT_URL:
                # Simulate API error or network error
                raise Exception("ColBERT down")
            elif url == WIKIPEDIA_API_URL:
                mock_wiki_resp = MagicMock()
                mock_wiki_resp.json.return_value = [  # type: ignore
                    query,
                    ["Wiki Title 1"],
                    ["Wiki Desc 1"],
                    ["http://wiki/1"],
                ]
                return mock_wiki_resp
            return MagicMock()

        mock_client.get.side_effect = side_effect  # type: ignore

        results = await fetch_colbert_results(query, k=1)

        assert len(results) == 1
        assert "Wiki Title 1" in results[0]["text"]
        assert results[0]["pid"] == "wiki-Wiki Title 1"
