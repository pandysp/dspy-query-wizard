import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import dspy  # type: ignore
from backend.app import app

@pytest.fixture
def mock_rag_modules():
    with patch("backend.app.HumanRAG") as MockHuman, \
         patch("backend.app.MachineRAG") as MockMachine:
        
        # Setup Human RAG mock
        human_instance = MagicMock()
        human_instance.return_value = dspy.Prediction(
            answer="Human Answer", 
            context=["Human Context"]
        )
        MockHuman.return_value = human_instance
        
        # Setup Machine RAG mock
        machine_instance = MagicMock()
        machine_instance.return_value = dspy.Prediction(
            answer="Machine Answer", 
            context=["Machine Context"],
            search_query="Machine Query"
        )
        MockMachine.return_value = machine_instance
        
        yield MockHuman, MockMachine

def test_query_endpoint(mock_rag_modules):
    """Test that the API endpoint calls both RAG pipelines and returns combined results."""
    MockHuman, MockMachine = mock_rag_modules
    
    with TestClient(app) as client:
        response = client.post("/api/query", json={"question": "Test Question"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Check structure
        assert data["question"] == "Test Question"
        assert data["human_answer"]["answer"] == "Human Answer"
        assert data["machine_answer"]["answer"] == "Machine Answer"
        assert data["machine_answer"]["search_query"] == "Machine Query"
        
        # Verify calls
        MockHuman.return_value.assert_called_with("Test Question")
        MockMachine.return_value.assert_called_with("Test Question")