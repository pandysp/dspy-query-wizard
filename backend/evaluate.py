import dspy  # type: ignore
import os
import json
import logging
from dspy.evaluate import Evaluate, answer_exact_match  # type: ignore
from dotenv import load_dotenv
from backend.rag import HumanRAG, MachineRAG

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

    # Import AgenticRAG dynamically to avoid circular dependency if any (though fine here)
    from backend.rag import AgenticRAG
    logger.info("Initializing AgenticRAG...")
    agentic_rag = AgenticRAG()
    compiled_agentic_path = os.path.join(os.path.dirname(__file__), "data", "compiled_agentic_rag.json")
    if os.path.exists(compiled_agentic_path):
        logger.info(f"Loading compiled AgenticRAG from {compiled_agentic_path}...")
        agentic_rag.load(compiled_agentic_path)
    else:
        logger.warning("No compiled AgenticRAG found! Running unoptimized.")

    # 3. Run Evaluation
    # dspy.Evaluate allows us to run metric over devset
    evaluator = Evaluate(devset=devset, metric=answer_exact_match, num_threads=1, display_progress=True, display_table=0)

    logger.info("\n--- Evaluating HumanRAG ---")
    human_score = evaluator(human_rag)
    
    logger.info("\n--- Evaluating MachineRAG ---")
    machine_score = evaluator(machine_rag)
    
    logger.info("\n--- Evaluating AgenticRAG ---")
    agentic_score = evaluator(agentic_rag)

    # 4. Scoreboard
    logger.info("\n" + "="*30)
    logger.info("FINAL SCOREBOARD")
    logger.info("="*30)
    logger.info(f"HumanRAG Accuracy:   {human_score}")
    logger.info(f"MachineRAG Accuracy: {machine_score}")
    logger.info(f"AgenticRAG Accuracy: {agentic_score}")
    logger.info("="*30)
    
    scores = {"Human": human_score, "Machine": machine_score, "Agentic": agentic_score}
    # Assuming EvaluationResult has a .score attribute or is float-like
    # If it's EvaluationResult, we need to extract score.
    # Based on previous run, it printed "3.0".
    # If it compares as numbers, good.
    # We'll try to find max.
    
    try:
        # Extract float if possible
        clean_scores = {}
        for k, v in scores.items():
            if hasattr(v, "score"):
                clean_scores[k] = v.score
            else:
                clean_scores[k] = v
        
        winner = max(clean_scores, key=clean_scores.get)
        logger.info(f"Winner: {winner}")
    except Exception:
        logger.info("Could not determine winner automatically.")

if __name__ == "__main__":
    evaluate()
