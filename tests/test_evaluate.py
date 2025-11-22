import pytest
from unittest.mock import MagicMock, patch, mock_open
import os

@pytest.mark.asyncio
async def test_evaluate_process():
    """Test the evaluation pipeline logic."""
    
    with (
        patch("backend.evaluate.Evaluate") as MockEvaluator,
        patch("backend.evaluate.dspy.Example") as _MockExample,
        patch(
            "builtins.open",
            mock_open(
                read_data='{"question": "q", "answer": "a", "supporting_facts": []}'
            ),
        ) as _mock_file,
                    patch("backend.evaluate.MachineRAG") as MockMachineRAG,
                    patch("backend.evaluate.HumanRAG") as MockHumanRAG,
                    patch("backend.rag.AgenticRAG") as MockAgenticRAG,
                    patch("os.path.exists") as mock_exists,        patch("backend.evaluate.configure_lm") as _mock_configure_lm,
    ):
        mock_exists.return_value = True
        
        # Setup Evaluator mock
        mock_evaluator_instance = MagicMock()
        MockEvaluator.return_value = mock_evaluator_instance
        # Return dummy scores
        mock_evaluator_instance.side_effect = [50.0, 80.0, 75.0] # Human, Machine, Agentic
        
        from backend.evaluate import evaluate
        
        # Run evaluation
        evaluate(sample_size=2)
        
        # Verify Evaluator initialized
        MockEvaluator.assert_called_once()
        
        # Verify all 3 pipelines evaluated
        assert mock_evaluator_instance.call_count == 3
