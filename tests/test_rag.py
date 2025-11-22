import pytest
from unittest.mock import MagicMock
import dspy  # type: ignore
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
    """Test the HumanRAG module's forward pass."""
    
    # Define a MockLM that inherits from dspy.LM to satisfy type checks
    class MockLM(dspy.LM):
        def __init__(self, responses):
            super().__init__("mock-model")
            self.responses = responses
            self.history = []

        def __call__(self, prompt=None, messages=None, **kwargs):
            # dspy.LM expects to return a list of strings (completions)
            return self.responses

    # 1. Setup Mocks
    
    # Mock Retriever
    mock_retriever = MagicMock()
    mock_retriever.return_value = dspy.Prediction(passages=["Paris is the capital of France."])
    
    # Mock LM
    # The ChainOfThought module expects "Reasoning: ... Answer: ..." usually
    # But if ChatAdapter fails, it falls back to JSONAdapter.
    # Let's provide JSON to be safe and robust.
    mock_lm = MockLM(['{"reasoning": "Because it is.", "answer": "Paris"}'])
    
    # 2. Initialize Module
    rag = HumanRAG(retriever=mock_retriever)
    
    # 3. Configure Global LM
    dspy.settings.configure(lm=mock_lm)
    
    # 4. Execute
    result = rag("What is the capital of France?")
    
    # 5. Assert
    assert result.answer == "Paris"
    assert result.context == ["Paris is the capital of France."]
    
    # Verify retriever was called
    mock_retriever.assert_called_once()