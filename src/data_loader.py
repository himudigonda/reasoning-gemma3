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
    logger.debug(
        f"data_loader.get_gsm8k_questions :: Entering function with split: {split}"
    )
    try:
        logger.info(
            f"data_loader.get_gsm8k_questions :: Loading dataset split: {split}"
        )
        data = load_dataset("openai/gsm8k", "main")[split]  # type: ignore
        logger.info("data_loader.get_gsm8k_questions :: Dataset loaded successfully")
    except ConnectionError as e:
        logger.error(
            f"data_loader.get_gsm8k_questions :: Failed to load dataset due to a connection error: {e}"
        )
        logger.debug(
            "data_loader.get_gsm8k_questions :: Exiting function with connection error"
        )
        return []
    except Exception as e:
        logger.error(
            f"data_loader.get_gsm8k_questions :: An unexpected error occurred while loading the dataset: {e}"
        )
        logger.debug(
            "data_loader.get_gsm8k_questions :: Exiting function with unexpected error"
        )
        return []

    def extract_hash_answer(text: str) -> str | None:
        logger.debug(
            "data_loader.get_gsm8k_questions.extract_hash_answer :: Entering inner function"
        )
        if "####" not in text:
            logger.debug(
                "data_loader.get_gsm8k_questions.extract_hash_answer :: '####' not found in text, returning None"
            )
            return None
        answer = text.split("####")[1].strip()
        logger.debug(
            f"data_loader.get_gsm8k_questions.extract_hash_answer :: Extracted answer: {answer}"
        )
        logger.debug(
            "data_loader.get_gsm8k_questions.extract_hash_answer :: Exiting inner function"
        )
        return answer

    logger.debug("data_loader.get_gsm8k_questions :: Mapping dataset")
    data = data.map(
        lambda x: {  # type: ignore
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )  # type: ignore
    logger.debug("data_loader.get_gsm8k_questions :: Dataset mapping complete")
    logger.debug(
        f"data_loader.get_gsm8k_questions :: Returning dataset with {len(data)} examples"
    )
    logger.debug("data_loader.get_gsm8k_questions :: Exiting function successfully")
    return data  # type: ignore


if __name__ == "__main__":
    # Example usage (for testing)
    logger.info("data_loader :: Running example usage")
    train_dataset = get_gsm8k_questions("train")
    if train_dataset:
        logger.info(f"data_loader :: Loaded {len(train_dataset)} training examples.")
    else:
        logger.error("data_loader :: Failed to load the training dataset.")
