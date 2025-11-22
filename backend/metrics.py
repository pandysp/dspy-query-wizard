import dspy  # type: ignore
import re
import string
from collections import Counter
import dspy # type: ignore

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculates word overlap F1 score."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def metric_answer_f1(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Wraps F1 score for DSPy evaluation."""
    return f1_score(pred.answer, example.answer)

def metric_recall_at_20(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    Calculates Recall@20.
    Checks what percentage of Gold Titles (from supporting_facts) appear in the retrieved context.
    Assumes the retriever has been configured to return k=20 passages.
    """
    # 1. Extract Gold Titles (HotPotQA format: [title, sent_id])
    # We use a set to count unique documents found
    gold_titles = set(item[0] for item in example.supporting_facts)
    if not gold_titles:
        return 1.0 # No facts needed?

    # 2. Check retrieved context
    # pred.context is a list of strings (passages)
    retrieved_passages = pred.context
    
    # Ensure we are actually evaluating @20 (or less if fewer results found)
    # This metric doesn't force the retrieval, it just measures what was retrieved.
    # It is the responsibility of the RAG module to call retrieve(k=20).
    
    found_count = 0
    
    for title in gold_titles:
        # Normalize title for matching
        norm_title = normalize_answer(title)
        for passage in retrieved_passages:
            if norm_title in normalize_answer(passage):
                found_count += 1
                break
    
    return found_count / len(gold_titles)

def metric_context_precision(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    Calculates Precision of the retrieved context.
    Precision = (Number of relevant passages) / (Total Number of Retrieved Passages)
    Relevant = contains a Gold Title.
    """
    gold_titles = set(item[0] for item in example.supporting_facts)
    if not gold_titles:
        return 0.0
        
    retrieved_passages = pred.context
    if not retrieved_passages:
        return 0.0
        
    found_count = 0
    for passage in retrieved_passages:
        # Check if this passage contains ANY gold title
        is_relevant = False
        for title in gold_titles:
            if normalize_answer(title) in normalize_answer(passage):
                is_relevant = True
                break
        if is_relevant:
            found_count += 1
            
    return found_count / len(retrieved_passages)

def combined_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    Average of Answer F1 and Recall@20.
    """
    f1 = metric_answer_f1(example, pred)
    recall = metric_recall_at_20(example, pred)
    return (f1 + recall) / 2.0

def metric_log_retrieval(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    Logs the top 5 retrieved snippets for inspection.
    Returns 0.0 (dummy value).
    """
    print(f"\nQuestion: {example.question}")
    print(f"Gold Answer: {example.answer}")
    print(f"Predicted Answer: {pred.answer}")
    print("Retrieved Contexts (Top 5):")
    for i, ctx in enumerate(pred.context[:5]):
        # Truncate for display
        snippet = ctx[:200] + "..." if len(ctx) > 200 else ctx
        print(f"  [{i+1}] {snippet}")
    return 0.0


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
