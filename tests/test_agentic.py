import pytest
from unittest.mock import MagicMock, patch
import dspy  # type: ignore
from backend.rag import AgenticRAG

@pytest.mark.asyncio
async def test_agentic_rag_initialization():
    """Test that AgenticRAG initializes dspy.ReAct with the correct tool."""
    
    with patch("backend.rag.dspy.ReAct") as MockReAct, \
         patch("backend.rag.retrieve") as mock_retrieve:
        
        rag = AgenticRAG()
        
        # Verify ReAct was initialized with the retrieve tool
        args, kwargs = MockReAct.call_args
        assert "tools" in kwargs
        tools = kwargs["tools"]
        assert tools[0] == mock_retrieve

@pytest.mark.asyncio
async def test_agentic_rag_forward():
    """Test AgenticRAG forward pass delegates to ReAct."""
    with patch("backend.rag.dspy.ReAct") as MockReAct:
        mock_react_instance = MagicMock()
        MockReAct.return_value = mock_react_instance
        
        mock_react_instance.return_value = dspy.Prediction(answer="Nolan")
        
        rag = AgenticRAG()
        result = rag("Question")
        
        assert result.answer == "Nolan"
        mock_react_instance.assert_called_once()

