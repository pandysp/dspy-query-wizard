import pytest
from unittest.mock import MagicMock, patch, mock_open
import os

@pytest.mark.asyncio
async def test_evaluate_process():
    """Test the evaluation pipeline logic."""
    
    # Import first to avoid breaking tiktoken/dspy imports with global open mock
    from backend.evaluate import evaluate
    
    with (
        patch("backend.evaluate.MachineRAG") as MockMachineRAG,
        patch("backend.evaluate.HumanRAG") as MockHumanRAG,
        patch("backend.rag.AgenticRAG") as MockAgenticRAG,
        patch("os.path.exists") as mock_exists,
        patch("backend.evaluate.configure_lm") as _mock_configure_lm,
        # Patch open only for the duration of the test logic
        patch(
            "builtins.open",
            mock_open(
                read_data='{"question": "q", "answer": "a", "supporting_facts": []}'
            ),
        ) as _mock_file,
    ):
        mock_exists.return_value = True
        
        # Setup RAG mocks to return dummy predictions
        MockHumanRAG.return_value.return_value = MagicMock(answer="Human", context=[])
        MockMachineRAG.return_value.return_value = MagicMock(answer="Machine", context=[], search_query="Query")
        MockAgenticRAG.return_value.return_value = MagicMock(answer="Agentic", history=[])
        
        # Run evaluation
        evaluate(sample_size=2)
        
        # Check if open was called with the analysis file
        write_calls = [call for call in _mock_file.mock_calls if "evaluation_analysis.json" in str(call)]
        assert len(write_calls) > 0, "Analysis file was not written"