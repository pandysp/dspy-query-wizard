import pytest
from unittest.mock import MagicMock
import dspy  # type: ignore
from backend.rag import HumanRAG, BasicQA, MachineRAG


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
    mock_lm = MockLM(['{"reasoning": "Because it is.", "answer": "Paris"}'])
    
    # 2. Initialize Module
    rag = HumanRAG(retriever=mock_retriever)
    
    # 3. Execute with Context
    with dspy.context(lm=mock_lm):
        result = rag("What is the capital of France?")
    
    # 4. Assert
    assert result.answer == "Paris"
    assert result.context == ["Paris is the capital of France."]
    
    # Verify retriever was called
    mock_retriever.assert_called_once()


@pytest.mark.asyncio
async def test_machine_rag_forward():
    """Test the MachineRAG module's forward pass."""
    
    # 1. Setup Mocks
    mock_retriever = MagicMock()
    mock_retriever.return_value = dspy.Prediction(passages=["Paris is the capital of France."])
    
    # Mock LM that responds based on the prompt instruction
    class SmartMockLM(dspy.LM):
        def __init__(self):
            super().__init__("mock-model")
            self.history = []

        def __call__(self, prompt=None, messages=None, **kwargs):
            # Determine which step we are in based on the prompt instructions
            # DSPy prompts usually include the signature instructions.
            
            # Ensure prompt is a string
            p_text = prompt if prompt else ""
            if messages:
                # Chat model usage, check messages
                for m in messages:
                    p_text += str(m)
            
            # Check keywords from docstrings
            if "Write a simple search query" in p_text:
                return ['{"reasoning": "Break down question.", "search_query": "capital of France"}']
            elif "Answer questions" in p_text:
                return ['{"reasoning": "Found it.", "answer": "Paris"}']
            else:
                # Fallback or error
                print(f"DEBUG: UNKNOWN PROMPT: {p_text[:100]}...")
                return ['{"reasoning": "Unknown", "answer": "Error"}']

    mock_lm = SmartMockLM()
    
    # 2. Initialize Module
    rag = MachineRAG(retriever=mock_retriever)
    
    # 3. Execute with Context
    with dspy.context(lm=mock_lm):
        result = rag("What is the capital of France?")
    
    # 4. Assert
    assert result.answer == "Paris"
    assert result.search_query == "capital of France"
    assert result.context == ["Paris is the capital of France."]
    
    # Verify retriever was called with the REPHRASED query
    mock_retriever.assert_called_with("capital of France")
