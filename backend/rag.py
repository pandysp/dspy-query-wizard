import dspy  # type: ignore
from typing import Any, cast
from backend.retriever import search_wikipedia


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
                context.extend(search_wikipedia(q, k=3))
        else:
            # Simple single query
            context = search_wikipedia(question, k=3)
        
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=str(prediction.answer))  # type: ignore


class AgenticSignature(dspy.Signature):  # type: ignore[misc]
    """
    Answer questions by using the 'search_wikipedia' tool to gather information.
    You MUST use the 'search_wikipedia' tool at least once. Do not answer from memory.
    Think step-by-step:
    1. Identify missing information.
    2. Use 'search_wikipedia' to find it.
    3. Formulate the answer based ONLY on the retrieved context.
    4. If the search results contain the answer, output the answer immediately. Do not search again for the same thing.
    """
    
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="the final answer to the question")


class AgenticRAG(dspy.Module):  # type: ignore[misc]
    """
    Agentic RAG pipeline using ReAct loop.
    Allows sequential tool calling and reasoning.
    """
    def __init__(self) -> None:
        super().__init__()
        self.react = dspy.ReAct(AgenticSignature, tools=[search_wikipedia])  # type: ignore
        
        # Add a manual demo to teach the format and force tool usage
        self.react.demos = [
            dspy.Example(
                question="Where was the director of Inception born?",
                history=[
                    "Thought: I need to find who directed Inception.",
                    "Action: search_wikipedia[Director of Inception]",
                    "Observation: ['Inception is a 2010 film directed by Christopher Nolan.']",
                    "Thought: Now I need to find where Christopher Nolan was born.",
                    "Action: search_wikipedia[Christopher Nolan birth place]",
                    "Observation: ['Christopher Nolan was born in Westminster, London.']",
                    "Thought: I have the answer.",
                ],
                answer="Westminster, London"
            ).with_inputs("question")
        ]

    def forward(self, question: str) -> dspy.Prediction:  # type: ignore[misc]
        prediction = self.react(question=question)  # type: ignore
        
        # Extract context from trajectory (ReAct history)
        context: list[str] = []
        # ReAct stores trace in 'trajectory'
        history = getattr(prediction, "trajectory", [])
        
        for step in history:
            # step is usually a string in ReAct history
            if isinstance(step, str) and step.startswith("Observation:"):
                # Extract content after "Observation:"
                content = step.replace("Observation:", "", 1).strip()
                # If content looks like a list string "['...']", try to clean it up slightly
                # but keeping raw content is safer than fragile parsing.
                context.append(content)
        
        return dspy.Prediction(
            answer=str(prediction.answer),
            history=history,
            context=context
        )