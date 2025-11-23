import dspy  # type: ignore
import os
import json
import logging
from dspy.teleprompt import MIPROv2  # type: ignore
from dspy.evaluate import answer_exact_match  # type: ignore
from dotenv import load_dotenv
from backend.rag import AgenticRAG
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
        logger.warning("OPENAI_API_KEY not found. DSPy optimization will likely fail.")
        return

    # Default to gpt-4o-mini as requested
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not model_name.startswith("openai/"):
        full_model_name = f"openai/{model_name}"
    else:
        full_model_name = model_name

    lm = dspy.LM(full_model_name, api_key=api_key)
    dspy.settings.configure(lm=lm)
    logger.info(f"LM configured: {full_model_name}")


def train(train_size: int = 5, val_size: int = 3) -> None:
    """
    Trains the AgenticRAG pipeline using MIPROv2.

    Args:
        train_size: Number of training examples to use (default: 5 for balanced optimization)
        val_size: Number of validation examples to use (default: 3 for balanced optimization)
    """
    configure_lm()

    # Configure a stronger teacher model for bootstrapping
    api_key = os.getenv("OPENAI_API_KEY")
    teacher_lm = dspy.LM("openai/gpt-5-mini", api_key=api_key)

    # Configure a prompt model for instruction generation (can use same as teacher)
    prompt_lm = dspy.LM("openai/gpt-5-mini", api_key=api_key)

    # 1. Load Training and Validation Data
    data_path = os.path.join(os.path.dirname(__file__), "data", "train.json")
    if not os.path.exists(data_path):
        logger.error(
            f"Training data not found at {data_path}. Run data_preprocess.py first."
        )
        return

    logger.info(f"Loading data from {data_path}...")
    trainset = []
    valset = []
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            # Read line by line or full json
            try:
                data = json.load(f)
                if isinstance(data, list):
                    raw_data = data
                else:
                    raw_data = [data]
            except json.JSONDecodeError:
                # Try line-json
                f.seek(0)
                raw_data = [json.loads(line) for line in f]

        # Convert to DSPy Examples
        # Split into train and validation sets
        total_needed = train_size + val_size
        for i, item in enumerate(raw_data[:total_needed]):
            example = dspy.Example(
                question=item["question"], answer=item["answer"]
            ).with_inputs("question")

            if i < train_size:
                trainset.append(example)
            else:
                valset.append(example)

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    logger.info(f"Loaded {len(trainset)} training examples and {len(valset)} validation examples.")

    # 2. Initialize Student (AgenticRAG)
    student = AgenticRAG()

    # 3. Define MIPROv2 Optimizer
    # Use "medium" auto setting for balanced optimization (<30min runtime)
    teleprompter = MIPROv2(
        metric=answer_in_context,
        prompt_model=prompt_lm,
        task_model=None,  # Will use the configured LM
        teacher_settings=dict(lm=teacher_lm),
        max_bootstrapped_demos=2,  # Slightly more bootstrapped examples
        max_labeled_demos=2,  # Slightly more labeled examples
        auto="medium",  # Balanced iterations for better optimization
        num_threads=20,  # Parallel evaluation
        verbose=True,
        track_stats=True,
    )

    # 4. Compile with MIPROv2
    logger.info("Starting MIPROv2 optimization with Teacher (gpt-5-mini)...")
    logger.info("This will optimize both instructions and few-shot examples...")
    try:
        compiled_rag = teleprompter.compile(
            student,
            trainset=trainset,
            valset=valset,
            # num_trials and other params set automatically by auto="medium"
            minibatch=True,
            minibatch_size=10,
            minibatch_full_eval_steps=3,
        )

        # 5. Save
        output_path = os.path.join(
            os.path.dirname(__file__), "data", "compiled_agentic_rag.json"
        )
        logger.info(f"Saving compiled program to {output_path}...")
        compiled_rag.save(output_path)
        logger.info("Training complete.")

    except Exception as e:
        logger.error(f"Compilation failed: {e}")


if __name__ == "__main__":
    train()
