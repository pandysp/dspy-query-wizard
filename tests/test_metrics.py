import pytest
from unittest.mock import MagicMock
import dspy # type: ignore
from backend.metrics import answer_in_context

def test_answer_in_context_success():
    example = MagicMock(answer="Paris")
    pred = MagicMock(context=["Paris is the capital of France."])
    assert answer_in_context(example, pred) is True

def test_answer_in_context_fail():
    example = MagicMock(answer="Paris")
    pred = MagicMock(context=["London is the capital of UK."])
    assert answer_in_context(example, pred) is False

def test_answer_in_context_case_insensitive():
    example = MagicMock(answer="paris")
    pred = MagicMock(context=["PARIS IS THE CAPITAL."])
    assert answer_in_context(example, pred) is True

def test_answer_in_context_empty():
    example = MagicMock(answer="Paris")
    pred = MagicMock(context=[])
    assert answer_in_context(example, pred) is False
