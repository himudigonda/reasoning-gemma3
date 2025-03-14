from datasets import load_dataset
import logging
from src.utils import extract_xml_answer
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def get_gsm8k_questions(split="train", model_name="google/gemma-3-1b-it") -> list[dict]:
    """Loads and pre-processes the GSM8k dataset."""
    logger.debug(
        f"data_loader.get_gsm8k_questions :: Entering function with split: {split}"
    )
    try:
        data = load_dataset("openai/gsm8k", "main")[split]  # type: ignore
    except ConnectionError as e:
        logger.error(
            f"data_loader.get_gsm8k_questions :: Failed to load dataset due to a connection error: {e}"
        )
        return []
    except Exception as e:
        logger.error(
            f"data_loader.get_gsm8k_questions :: An unexpected error occurred while loading the dataset: {e}"
        )
        return []

    data = data.map(
        lambda x: {  # type: ignore
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )  # type: ignore
    return data  # type: ignore


if __name__ == "__main__":
    # Example usage (for testing)
    logger.info("data_loader :: Running example usage")
    train_dataset = get_gsm8k_questions("train")
    if train_dataset:
        logger.info(f"data_loader :: Loaded {len(train_dataset)} training examples.")
    else:
        logger.error("data_loader :: Failed to load the training dataset.")
