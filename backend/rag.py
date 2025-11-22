import dspy  # type: ignore
from typing import Any, cast
from backend.retriever import retrieve


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

    def __init__(self) -> None:
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(BasicQA)  # type: ignore

    def forward(self, question: str, queries: list[str] | None = None) -> dspy.Prediction:  # type: ignore[misc]
        # Use functional retrieval (returns list[str])
        context = []
        if queries:
            # Manual multi-query (Human simulated effort)
            for q in queries:
                context.extend(retrieve(q, k=3))
        else:
            # Simple single query
            context = retrieve(question, k=3)
        
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=str(prediction.answer))  # type: ignore


class GenerateSearchQuery(dspy.Signature):  # type: ignore[misc]
    """Write a simple search query that will help answer a complex question."""

    question: Any = dspy.InputField()
    search_query: Any = dspy.OutputField(desc="a simple keyword search query")


class MachineRAG(dspy.Module):  # type: ignore[misc]
    """
    Optimized RAG pipeline (Machine approach).
    Rephrases the question into a search query, retrieves, then answers.
    """

    def __init__(self) -> None:
        super().__init__()
        self.rephrase = dspy.ChainOfThought(GenerateSearchQuery)  # type: ignore
        self.generate_answer = dspy.ChainOfThought(BasicQA)  # type: ignore

    def forward(self, question: str) -> dspy.Prediction:  # type: ignore[misc]
        # 1. Rephrase
        rephrased = self.rephrase(question=question)
        raw_query = rephrased.search_query
        
        # Robustly handle potential non-string outputs from LM (e.g. JSON objects/lists)
        if isinstance(raw_query, dict):
            # Join values or take first value
            search_query = " ".join(str(v) for v in raw_query.values())
        elif isinstance(raw_query, list):
            search_query = " ".join(str(x) for x in raw_query)
        else:
            search_query = str(raw_query)

        # 2. Retrieve (functional)
        # Returns list[str] directly
        context = retrieve(search_query, k=3)

        # 3. Answer
        prediction = self.generate_answer(context=context, question=question)

        return dspy.Prediction(
            context=context, answer=str(prediction.answer), search_query=search_query
        )  # type: ignore


class AgenticSignature(dspy.Signature):  # type: ignore[misc]
    """Answer complex questions by using search tools to gather information step by step."""
    
    question: Any = dspy.InputField()
    answer: Any = dspy.OutputField(desc="the final answer to the question")


class AgenticRAG(dspy.Module):  # type: ignore[misc]
    """
    Agentic RAG pipeline using ReAct loop.
    Allows sequential tool calling and reasoning.
    """
    def __init__(self) -> None:
        super().__init__()
        self.react = dspy.ReAct(AgenticSignature, tools=[retrieve])  # type: ignore

    def forward(self, question: str) -> dspy.Prediction:  # type: ignore[misc]
        prediction = self.react(question=question)  # type: ignore
        
        # Extract context from history observations
        context: list[str] = []
        history = getattr(prediction, "history", [])
        
        for step in history:
            # step is usually a string in ReAct history
            if isinstance(step, str) and step.startswith("Observation:"):
                # Extract content after "Observation:"
                content = step.replace("Observation:", "", 1).strip()
                # If content looks like a list string "['...']", try to clean it up slightly
                # but keeping raw content is safer than fragile parsing.
                context.append(content)
        
        return dspy.Prediction(
            answer=prediction.answer,
            history=history,
            context=context
        )