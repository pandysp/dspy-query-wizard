import dspy  # type: ignore
from typing import Any


class BasicQA(dspy.Signature):  # type: ignore[misc]
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class HumanRAG(dspy.Module):  # type: ignore[misc]
    """
    Standard RAG pipeline (Human approach).
    Retrieves context then answers the question.
    """

    def __init__(self, retriever: dspy.Module | None = None) -> None:
        super().__init__()
        # If no retriever provided, use the global one (dspy.Retrieve)
        # We pass k=3 to match our plan constraints
        self.retrieve = retriever if retriever else dspy.Retrieve(k=3)  # type: ignore
        self.generate_answer = dspy.ChainOfThought(BasicQA)  # type: ignore

    def forward(self, question: str) -> dspy.Prediction:  # type: ignore[misc]
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)  # type: ignore
