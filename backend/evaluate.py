import dspy  # type: ignore
import os
import json
import logging
from dspy.evaluate import answer_exact_match  # type: ignore
from dotenv import load_dotenv
from backend.rag import HumanRAG, MachineRAG
from backend.metrics import answer_in_context

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def configure_lm() -> None:
    """Configures the Language Model."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found. Evaluation will likely fail.")
        return

    model_name = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    if not model_name.startswith("openai/"):
        full_model_name = f"openai/{model_name}"
    else:
        full_model_name = model_name

    lm = dspy.LM(full_model_name, api_key=api_key)
    dspy.settings.configure(lm=lm)
    logger.info(f"LM configured: {full_model_name}")

def evaluate(sample_size: int = 10) -> None:
    """
    Evaluates HumanRAG vs MachineRAG vs AgenticRAG on the eval split.
    Collects detailed traces and saves them to 'backend/data/evaluation_analysis.json'.
    """
    configure_lm()

    # 1. Load Eval Data
    data_path = os.path.join(os.path.dirname(__file__), "data", "eval.json")
    if not os.path.exists(data_path):
        logger.error(f"Eval data not found at {data_path}. Run data_preprocess.py first.")
        return

    logger.info(f"Loading eval data from {data_path}...")
    devset = []
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    raw_data = data
                else:
                    raw_data = [data]
            except json.JSONDecodeError:
                f.seek(0)
                raw_data = [json.loads(line) for line in f]

        for item in raw_data[:sample_size]:
            example = dspy.Example(
                question=item["question"],
                answer=item["answer"]
            ).with_inputs("question")
            devset.append(example)
            
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    logger.info(f"Loaded {len(devset)} eval examples.")

    # 2. Initialize Pipelines
    logger.info("Initializing HumanRAG...")
    human_rag = HumanRAG()
    
    logger.info("Initializing MachineRAG...")
    machine_rag = MachineRAG()
    compiled_path = os.path.join(os.path.dirname(__file__), "data", "compiled_machine_rag.json")
    if os.path.exists(compiled_path):
        logger.info(f"Loading compiled MachineRAG from {compiled_path}...")
        machine_rag.load(compiled_path)
    else:
        logger.warning("No compiled MachineRAG found! Running unoptimized.")

    from backend.rag import AgenticRAG
    logger.info("Initializing AgenticRAG...")
    agentic_rag = AgenticRAG()
    compiled_agentic_path = os.path.join(os.path.dirname(__file__), "data", "compiled_agentic_rag.json")
    if os.path.exists(compiled_agentic_path):
        logger.info(f"Loading compiled AgenticRAG from {compiled_agentic_path}...")
        agentic_rag.load(compiled_agentic_path)
    else:
        logger.warning("No compiled AgenticRAG found! Running unoptimized.")

    # 3. Custom Evaluation Loop
    results = []
    # Scores: { "Human": {"acc": 0, "recall": 0}, ... }
    metrics = {
        "Human": {"acc": 0, "recall": 0},
        "Machine": {"acc": 0, "recall": 0},
        "Agentic": {"acc": 0, "recall": 0}
    }
    
    logger.info("\n--- Starting Evaluation Loop ---")
    
    for i, example in enumerate(devset):
        question = example.question
        gold_answer = example.answer
        
        logger.info(f"[{i+1}/{len(devset)}] Q: {question}")
        
        # Helper to run and eval
        def run_and_eval(pipeline, name):
            try:
                pred = pipeline(question)
                correct = answer_exact_match(example, pred)
                recall = answer_in_context(example, pred)
                
                if correct: metrics[name]["acc"] += 1
                if recall: metrics[name]["recall"] += 1
                
                return pred, correct, recall
            except Exception as e:
                logger.error(f"{name} failed: {e}")
                # Return dummy prediction
                return dspy.Prediction(answer="Error", context=[], search_query="Error", history=[]), False, False

        human_pred, human_correct, human_recall = run_and_eval(human_rag, "Human")
        machine_pred, machine_correct, machine_recall = run_and_eval(machine_rag, "Machine")
        agentic_pred, agentic_correct, agentic_recall = run_and_eval(agentic_rag, "Agentic")
            
        # Collect Data
        result_entry = {
            "question": question,
            "gold_answer": gold_answer,
            "human": {
                "answer": human_pred.answer,
                "correct": human_correct,
                "recall": human_recall,
                "context_sample": human_pred.context[:1] if hasattr(human_pred, "context") and human_pred.context else []
            },
            "machine": {
                "answer": machine_pred.answer,
                "correct": machine_correct,
                "recall": machine_recall,
                "search_query": getattr(machine_pred, "search_query", None),
                "context_sample": machine_pred.context[:1] if hasattr(machine_pred, "context") and machine_pred.context else []
            },
            "agentic": {
                "answer": agentic_pred.answer,
                "correct": agentic_correct,
                "recall": agentic_recall,
                "trace": getattr(agentic_pred, "history", [])
            }
        }
        results.append(result_entry)

    # 4. Save Artifacts
    analysis_path = os.path.join(os.path.dirname(__file__), "data", "evaluation_analysis.json")
    logger.info(f"Saving analysis to {analysis_path}...")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 5. Scoreboard
    def get_pct(val): return (val / len(devset)) * 100

    logger.info("\n" + "="*40)
    logger.info("FINAL SCOREBOARD")
    logger.info("="*40)
    logger.info(f"{'Pipeline':<10} | {'Recall (Ret)':<13} | {'Accuracy':<10}")
    logger.info("-" * 45)
    
    for name, scores in metrics.items():
        recall_pct = get_pct(scores["recall"])
        acc_pct = get_pct(scores["acc"])
        logger.info(f"{name:<10} | {recall_pct:<12.1f}% | {acc_pct:<9.1f}%")
    
    logger.info("="*45)
    
    # Determine Winner based on Recall
    winner = max(metrics, key=lambda k: metrics[k]["recall"])
    logger.info(f"Winner (Retrieval Accuracy): {winner}")

if __name__ == "__main__":
    evaluate()