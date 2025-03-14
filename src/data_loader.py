from datasets import load_dataset
import logging
from src.utils import extract_xml_answer

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


def get_gsm8k_questions(split="train") -> list[dict]:
    """Loads and pre-processes the GSM8k dataset."""
    try:
        data = load_dataset("openai/gsm8k", "main")[split]  # type: ignore
    except ConnectionError as e:
        logger.error(f"Failed to load dataset due to a connection error: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the dataset: {e}")
        return []

    def extract_hash_answer(text: str) -> str | None:
        if "####" not in text:
            return None
        return text.split("####")[1].strip()

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
    train_dataset = get_gsm8k_questions("train")
    if train_dataset:
        print(f"Loaded {len(train_dataset)} training examples.")
    else:
        print("Failed to load the training dataset.")
