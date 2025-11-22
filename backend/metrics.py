import dspy  # type: ignore

def answer_in_context(example, pred, trace=None):
    """
    Returns True if the gold answer string appears in the retrieved context.
    Checks if 'answer' (from example) is a substring of any passage in 'context' (from pred).
    Case insensitive.
    """
    # Extract answer
    # dspy.Example usually has attributes accessed directly
    answer = example.answer
    if not answer:
        return False
        
    # Extract context
    # Prediction usually has 'context' which is list[str]
    context = getattr(pred, "context", [])
    if not context:
        return False
    
    # Normalize
    answer_norm = str(answer).lower().strip()
    
    # Check
    # We join context to search across boundaries? No, usually passage based.
    # Let's search in full text to be safe.
    full_text = " ".join(context).lower()
    
    return answer_norm in full_text
