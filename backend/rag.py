import dspy  # type: ignore
from typing import Any, cast


class BasicQA(dspy.Signature):  # type: ignore[misc]
    """Answer questions with short factoid answers."""

    context: Any = dspy.InputField(desc="may contain relevant facts")
    question: Any = dspy.InputField()
    answer: Any = dspy.OutputField(desc="often between 1 and 5 words")


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
        # Retrieve returns a Prediction which we access passages from
        retrieved: Any = self.retrieve(question)
        context = cast(list[str], retrieved.passages)

        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)  # type: ignore


class GenerateSearchQuery(dspy.Signature):  # type: ignore[misc]
    """Write a simple search query that will help answer a complex question."""

    question: Any = dspy.InputField()
    search_query: Any = dspy.OutputField(desc="a simple keyword search query")


class MachineRAG(dspy.Module):  # type: ignore[misc]
    """
    Optimized RAG pipeline (Machine approach).
    Rephrases the question into a search query, retrieves, then answers.
    """

    def __init__(self, retriever: dspy.Module | None = None) -> None:
        super().__init__()
        self.retrieve = retriever if retriever else dspy.Retrieve(k=3)  # type: ignore
        self.rephrase = dspy.ChainOfThought(GenerateSearchQuery)  # type: ignore
        self.generate_answer = dspy.ChainOfThought(BasicQA)  # type: ignore

    def forward(self, question: str) -> dspy.Prediction:  # type: ignore[misc]
        # 1. Rephrase
        rephrased = self.rephrase(question=question)
        search_query = cast(str, rephrased.search_query)

        # 2. Retrieve (using rephrased query)
        retrieved: Any = self.retrieve(search_query)
        context = cast(list[str], retrieved.passages)

        # 3. Answer
        prediction = self.generate_answer(context=context, question=question)

        return dspy.Prediction(
            context=context, answer=prediction.answer, search_query=search_query
        )  # type: ignore
