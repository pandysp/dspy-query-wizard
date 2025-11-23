import dspy  # type: ignore
import os
import json
import logging
from dspy.evaluate import answer_exact_match, Evaluate  # type: ignore
from dotenv import load_dotenv
from backend.rag import HumanRAG, AgenticRAG
from backend.metrics import answer_in_context, metric_recall_at_20, metric_log_retrieval

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

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not model_name.startswith("openai/"):
        full_model_name = f"openai/{model_name}"
    else:
        full_model_name = model_name

    lm = dspy.LM(full_model_name, api_key=api_key)
    dspy.settings.configure(lm=lm)
    logger.info(f"LM configured: {full_model_name}")

def evaluate(sample_size: int = 10) -> None:
    """
    Evaluates HumanRAG vs AgenticRAG on the eval split.
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
            # Convert supporting_facts from columnar to row format
            supporting_facts = []
            raw_facts = item.get("supporting_facts", {})
            if raw_facts and "title" in raw_facts and "sent_id" in raw_facts:
                titles = raw_facts["title"]
                sent_ids = raw_facts["sent_id"]
                supporting_facts = list(zip(titles, sent_ids))

            example = dspy.Example(
                question=item["question"],
                answer=item["answer"],
                supporting_facts=supporting_facts
            ).with_inputs("question")
            devset.append(example)
            
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    logger.info(f"Loaded {len(devset)} eval examples.")

    # 2. Initialize Pipelines
    logger.info("Initializing HumanRAG...")
    human_rag = HumanRAG()

    logger.info("Initializing AgenticRAG...")
    agentic_rag = AgenticRAG()
    compiled_agentic_path = os.path.join(os.path.dirname(__file__), "data", "compiled_agentic_rag.json")
    if os.path.exists(compiled_agentic_path):
        logger.info(f"Loading compiled AgenticRAG from {compiled_agentic_path}...")
        agentic_rag.load(compiled_agentic_path)
    else:
        logger.warning("No compiled AgenticRAG found! Running unoptimized.")

    # 3. Run Evaluation

    # --- Inspection Phase ---
    logger.info("\n" + "="*30)
    logger.info("INSPECTION PHASE (First 3 Examples)")
    logger.info("="*30)
    
    inspector = Evaluate(devset=devset[:3], metric=metric_log_retrieval, num_threads=1, display_progress=True, display_table=0)
    
    logger.info("\n--- Inspecting HumanRAG ---")
    inspector(human_rag)

    logger.info("\n--- Inspecting AgenticRAG ---")
    inspector(agentic_rag)

    # --- Accuracy Phase ---
    logger.info("\n" + "="*30)
    logger.info("ACCURACY PHASE (Exact Match)")
    logger.info("="*30)
    
    evaluator_em = Evaluate(devset=devset, metric=answer_exact_match, num_threads=1, display_progress=True, display_table=0)

    logger.info("\n--- Evaluating HumanRAG (Accuracy) ---")
    human_em = evaluator_em(human_rag)
    
    logger.info("\n--- Evaluating AgenticRAG (Accuracy) ---")
    agentic_em = evaluator_em(agentic_rag)

    # --- Recall Phase ---
    logger.info("\n" + "="*30)
    logger.info("RECALL PHASE (Recall@20)")
    logger.info("="*30)
    
    evaluator_recall = Evaluate(devset=devset, metric=metric_recall_at_20, num_threads=1, display_progress=True, display_table=0)

    logger.info("\n--- Evaluating HumanRAG (Recall) ---")
    human_recall = evaluator_recall(human_rag)
    
    logger.info("\n--- Evaluating AgenticRAG (Recall) ---")
    agentic_recall = evaluator_recall(agentic_rag)

    # 4. Scoreboard
    logger.info("\n" + "="*30)
    logger.info("FINAL SCOREBOARD")
    logger.info("="*30)
    logger.info(f"{'Model':<15} | {'Accuracy':<10} | {'Recall@20':<10}")
    logger.info("-" * 41)
    
    def get_score(res):
        # Handle dspy.EvaluationResult object if present
        if hasattr(res, "score"):
            return res.score
        return res

    logger.info(f"{'HumanRAG':<15} | {get_score(human_em):<10.2f} | {get_score(human_recall):<10.2f}")
    logger.info(f"{'AgenticRAG':<15} | {get_score(agentic_em):<10.2f} | {get_score(agentic_recall):<10.2f}")
    logger.info("="*30)

if __name__ == "__main__":
    evaluate()