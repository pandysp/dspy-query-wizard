import pytest
from unittest.mock import MagicMock, patch
import dspy  # type: ignore
from backend.rag import AgenticRAG

@pytest.mark.asyncio
async def test_agentic_rag_initialization():
    """Test that AgenticRAG initializes dspy.ReAct with the correct tool."""
    
    with patch("backend.rag.dspy.ReAct") as MockReAct, \
         patch("backend.rag.search_wikipedia") as mock_retrieve:
        
        rag = AgenticRAG()
        
        # Verify ReAct was initialized with the retrieve tool
        args, kwargs = MockReAct.call_args
        assert "tools" in kwargs
        tools = kwargs["tools"]
        assert tools[0] == mock_retrieve

@pytest.mark.asyncio
async def test_agentic_rag_forward():
    """Test AgenticRAG forward pass delegates to ReAct and extracts context from trajectory."""
    with patch("backend.rag.dspy.ReAct") as MockReAct:
        mock_react_instance = MagicMock()
        MockReAct.return_value = mock_react_instance
        
        # Mock ReAct output with trajectory (ReAct specific field)
        mock_react_instance.return_value = dspy.Prediction(
            answer="Nolan",
            trajectory=[
                "Thought: Find director.",
                "Action: search_wikipedia('Inception')",
                "Observation: ['Inception is directed by Christopher Nolan.']",
                "Thought: Answer found."
            ]
        )
        
        rag = AgenticRAG()
        result = rag("Question")
        
        assert result.answer == "Nolan"
        # Verify context extraction from trajectory
        assert "Inception is directed by Christopher Nolan." in str(result.context)
        mock_react_instance.assert_called_once()

