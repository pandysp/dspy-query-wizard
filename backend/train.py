import dspy  # type: ignore
import os
import json
import logging
from dspy.teleprompt import BootstrapFewShot  # type: ignore
from dspy.evaluate import answer_exact_match  # type: ignore
from backend.rag import MachineRAG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(sample_size: int = 20) -> None:
    """
    Trains the MachineRAG pipeline using BootstrapFewShot.
    """
    # 1. Load Training Data
    data_path = os.path.join(os.path.dirname(__file__), "data", "train.json")
    if not os.path.exists(data_path):
        logger.error(
            f"Training data not found at {data_path}. Run data_preprocess.py first."
        )
        return

    logger.info(f"Loading training data from {data_path}...")
    trainset = []
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
        for item in raw_data[:sample_size]:
            # HotPotQA structure usually has question, answer, supporting_facts
            # We need to map to our signature inputs: 'question'
            # And provide labels for metric: 'answer'
            example = dspy.Example(
                question=item["question"], answer=item["answer"]
            ).with_inputs("question")
            trainset.append(example)

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    logger.info(f"Loaded {len(trainset)} examples.")

    # 2. Initialize Student (MachineRAG)
    # We use the default retriever (ColBERTv2)
    student = MachineRAG()

    # 3. Define Optimizer
    # We optimize for exact match on the final answer
    teleprompter = BootstrapFewShot(
        metric=answer_exact_match, max_bootstrapped_demos=4, max_labeled_demos=4
    )

    # 4. Compile
    logger.info("Starting compilation...")
    compiled_rag = teleprompter.compile(student, trainset=trainset)

    # 5. Save
    output_path = os.path.join(
        os.path.dirname(__file__), "data", "compiled_machine_rag.json"
    )
    logger.info(f"Saving compiled program to {output_path}...")
    compiled_rag.save(output_path)
    logger.info("Training complete.")


if __name__ == "__main__":
    # Ensure DSPy is configured (usually done in app.py or by environment)
    # For standalone script, we might need to configure it.
    # dspy.settings.configure(lm=..., rm=...)
    train()
