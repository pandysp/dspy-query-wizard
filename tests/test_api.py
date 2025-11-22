import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import dspy  # type: ignore
from backend.app import app


@pytest.fixture
def mock_rag_modules():
    with (
        patch("backend.app.HumanRAG") as MockHuman,
        patch("backend.app.AgenticRAG") as MockAgentic,
    ):
        # Setup Human RAG mock
        human_instance = MagicMock()
        human_instance.return_value = dspy.Prediction(
            answer="Human Answer", context=["Human Context"]
        )
        MockHuman.return_value = human_instance

        # Setup Agentic RAG mock
        agentic_instance = MagicMock()
        agentic_instance.return_value = dspy.Prediction(
            answer="Agentic Answer",
            history=["Step 1", "Step 2"]
        )
        MockAgentic.return_value = agentic_instance

        yield MockHuman, MockAgentic


def test_query_endpoint(mock_rag_modules):
    """Test that the API endpoint calls both RAG pipelines and returns combined results."""
    MockHuman, MockAgentic = mock_rag_modules

    with TestClient(app) as client:
        # Test with manual queries
        response = client.post(
            "/api/query",
            json={
                "question": "Test Question",
                "manual_queries": ["query1", "query2"]
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Check structure
        assert data["question"] == "Test Question"
        assert data["human_answer"]["answer"] == "Human Answer"
        assert data["agentic_answer"]["answer"] == "Agentic Answer"

        # Verify calls
        # HumanRAG should receive the manual queries
        MockHuman.return_value.assert_called_with("Test Question", queries=["query1", "query2"])
        MockAgentic.return_value.assert_called_with("Test Question")
