import pytest
from unittest.mock import MagicMock, patch, mock_open
import dspy # type: ignore
import os

@pytest.mark.asyncio
async def test_train_process():
    """Test the training pipeline logic."""
    
    # Mock dependencies
    with patch("backend.train.BootstrapFewShot") as MockOptimizer, \
         patch("backend.train.dspy.Example") as MockExample, \
         patch("builtins.open", mock_open(read_data='{"question": "q", "answer": "a", "supporting_facts": []}')) as mock_file, \
         patch("backend.train.MachineRAG") as MockMachineRAG, \
         patch("os.path.exists") as mock_exists:
        
        mock_exists.return_value = True
        
        # Setup mocks
        mock_optimizer_instance = MagicMock()
        MockOptimizer.return_value = mock_optimizer_instance
        
        # Configure MachineRAG mock to pass DSPy checks
        mock_machine_rag_instance = MagicMock()
        MockMachineRAG.return_value = mock_machine_rag_instance
        # Ensure _compiled is False so assertion passes
        mock_machine_rag_instance._compiled = False
        # Ensure copies also look uncompiled
        mock_machine_rag_instance.reset_copy.return_value = mock_machine_rag_instance
        mock_machine_rag_instance.deepcopy.return_value = mock_machine_rag_instance
        
        mock_compiled_program = MagicMock()
        mock_optimizer_instance.compile.return_value = mock_compiled_program
        
        # Import here to avoid early import issues if we rely on global state
        from backend.train import train
        
        # Run training
        # We pass a dummy path or ensure it uses a default
        train(sample_size=2)
        
        # Verify optimizer was initialized
        MockOptimizer.assert_called_once()
        
        # Verify compile was called
        mock_optimizer_instance.compile.assert_called_once()
        
        # Verify save was called
        mock_compiled_program.save.assert_called_once()