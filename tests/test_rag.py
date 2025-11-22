import pytest
from unittest.mock import MagicMock, patch
import dspy  # type: ignore
import backend.rag
from backend.rag import HumanRAG, BasicQA


@pytest.mark.asyncio
async def test_human_rag_signature():
    """Test that the BasicQA signature has the correct fields."""
    # Access class attributes directly for dspy.Signature
    assert "context" in BasicQA.input_fields
    assert "question" in BasicQA.input_fields
    assert "answer" in BasicQA.output_fields


@pytest.mark.asyncio
async def test_human_rag_forward():
    """Test the HumanRAG module's forward pass using functional retrieval."""
    
    # Define a MockLM
    class MockLM(dspy.LM):
        def __init__(self, responses):
            super().__init__("mock-model")
            self.responses = responses
            self.history = []

        def __call__(self, prompt=None, messages=None, **kwargs):
            return self.responses

    # Mock LM
    mock_lm = MockLM(['{"reasoning": "Because it is.", "answer": "Paris"}'])
    
    with patch("backend.rag.search_wikipedia") as mock_retrieve:
        mock_retrieve.return_value = ["Paris is the capital of France."]
        
        rag = HumanRAG()
        
        with dspy.context(lm=mock_lm):
            result = rag("What is the capital of France?")
        
        assert result.answer == "Paris"
        assert result.context == ["Paris is the capital of France."]
        
        mock_retrieve.assert_called_once_with("What is the capital of France?", k=3)

@pytest.mark.asyncio
async def test_human_rag_manual_queries():
    """Test HumanRAG with manually provided search queries."""
    
    class MockLM(dspy.LM):
        def __init__(self, responses):
            super().__init__("mock-model")
            self.responses = responses
            self.history = []
        def __call__(self, prompt=None, messages=None, **kwargs):
            return self.responses

    mock_lm = MockLM(['{"reasoning": "Combined info.", "answer": "Paris"}'])
    
    with patch("backend.rag.search_wikipedia") as mock_retrieve:
        # Mock returning different contexts for different queries
        def side_effect(query, k=3):
            if "query1" in query:
                return ["Context 1"]
            if "query2" in query:
                return ["Context 2"]
            return []
        mock_retrieve.side_effect = side_effect
        
        rag = HumanRAG()
        
        with dspy.context(lm=mock_lm):
            # Pass manual queries
            result = rag("Complex Question", queries=["query1", "query2"])
        
        assert result.answer == "Paris"
        # Should contain both contexts
        assert "Context 1" in result.context
        assert "Context 2" in result.context
        
        assert mock_retrieve.call_count == 2


