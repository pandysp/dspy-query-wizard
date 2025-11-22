import pytest
from unittest.mock import MagicMock, patch, mock_open
import dspy # type: ignore
import os

@pytest.mark.asyncio
async def test_train_agentic_process():
    """Test the agentic training pipeline logic."""
    
    # Mock dependencies
    with (
        patch("backend.train_agentic.BootstrapFewShot") as MockOptimizer,
        patch("backend.train_agentic.dspy.Example") as _MockExample,
        patch(
            "builtins.open",
            mock_open(
                read_data='{"question": "q", "answer": "a", "supporting_facts": []}'
            ),
        ) as _mock_file,
        patch("backend.train_agentic.AgenticRAG") as MockAgenticRAG,
        patch("os.path.exists") as mock_exists,
        patch("backend.train_agentic.configure_lm") as _mock_configure_lm,
    ):
        mock_exists.return_value = True
        
        # Setup mocks
        mock_optimizer_instance = MagicMock()
        MockOptimizer.return_value = mock_optimizer_instance
        
        mock_agentic_instance = MagicMock()
        MockAgenticRAG.return_value = mock_agentic_instance
        # Configure mock to look uncompiled
        mock_agentic_instance._compiled = False
        mock_agentic_instance.reset_copy.return_value = mock_agentic_instance
        mock_agentic_instance.deepcopy.return_value = mock_agentic_instance
        
        mock_compiled_program = MagicMock()
        mock_optimizer_instance.compile.return_value = mock_compiled_program
        
        # Import
        from backend.train_agentic import train
        
        # Run training
        train(sample_size=2)
        
        # Verify optimizer was initialized
        MockOptimizer.assert_called_once()
        
        # Verify compile was called
        mock_optimizer_instance.compile.assert_called_once()
        
        # Verify save was called
        mock_compiled_program.save.assert_called_once()
