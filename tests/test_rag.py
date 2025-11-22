import pytest
from unittest.mock import patch
import dspy  # type: ignore
# We will now depend on the module importing 'retrieve', so we patch it where it is used.
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
    
    # Mock the functional 'retrieve' imported in backend.rag
    # We assume backend.rag imports 'retrieve' from backend.retriever
    with patch("backend.rag.retrieve") as mock_retrieve:
        mock_retrieve.return_value = ["Paris is the capital of France."]
        
        # 2. Initialize Module (No retriever arg needed anymore)
        rag = HumanRAG()
        
        # 3. Execute with Context
        with dspy.context(lm=mock_lm):
            result = rag("What is the capital of France?")
        
        # 4. Assert
        assert result.answer == "Paris"
        assert result.context == ["Paris is the capital of France."]
        
        # Verify functional retriever was called
        mock_retrieve.assert_called_once_with("What is the capital of France?", k=3)


    @pytest.mark.asyncio
    async def test_machine_rag_forward():
        """Test the MachineRAG module's forward pass using functional retrieval."""
        
        # Mock LM    
    class SmartMockLM(dspy.LM):
        def __init__(self):
            super().__init__("mock-model")
            self.history = []

        def __call__(self, prompt=None, messages=None, **kwargs):
            p_text = prompt if prompt else ""
            if messages:
                for m in messages:
                    p_text += str(m)
            
            if "simple search query" in p_text or "Search Query" in p_text:
                return ['{"reasoning": "Break down question.", "search_query": "capital of France"}']
            elif "Answer questions" in p_text or "Answer:" in p_text:
                return ['{"reasoning": "Found it.", "answer": "Paris"}']
            
            return ['{"answer": "Error"}']

    mock_lm = SmartMockLM()
    
    with patch("backend.rag.retrieve") as mock_retrieve:
        mock_retrieve.return_value = ["Paris is the capital of France."]
        
        # 2. Initialize Module
        rag = MachineRAG()
        
        # 3. Execute with Context
        with dspy.context(lm=mock_lm):
            result = rag("What is the capital of France?")
        
        # 4. Assert
        assert result.answer == "Paris"
        assert result.search_query == "capital of France"
        assert result.context == ["Paris is the capital of France."]
        
        # Verify retriever was called with the REPHRASED query
        mock_retrieve.assert_called_with("capital of France", k=3)
