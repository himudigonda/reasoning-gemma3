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


def get_gsm8k_questions(
    model_name, max_prompt_length, max_completion_length, split="train"
) -> list[dict]:
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

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token_id = 0  # Ensure pad token is set

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

    def preprocess_function(examples):
        prompts = [SYSTEM_PROMPT + "\n" + example["question"] for example in examples]
        answers = [extract_hash_answer(example["answer"]) for example in examples]  # type: ignore

        # Tokenize prompts and answers
        prompt_tokenized = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_prompt_length,
            return_tensors="pt",
        )
        answer_tokenized = tokenizer(answers, padding="max_length", truncation=True, max_length=max_completion_length, return_tensors="pt")  # type: ignore

        return {
            "prompt": prompt_tokenized["input_ids"],
            "answer": answer_tokenized["input_ids"],
            "attention_mask": prompt_tokenized["attention_mask"],
        }

    logger.debug("data_loader.get_gsm8k_questions :: Mapping dataset")
    processed_data = data.map(
        preprocess_function,
        batched=True,
        remove_columns=data.column_names,
    )
    logger.debug("data_loader.get_gsm8k_questions :: Dataset mapping complete")
    logger.debug(
        f"data_loader.get_gsm8k_questions :: Returning dataset with {len(processed_data)} examples"
    )
    logger.debug("data_loader.get_gsm8k_questions :: Exiting function successfully")
    return processed_data


if __name__ == "__main__":
    # Example usage (for testing)
    logger.info("data_loader :: Running example usage")
    train_dataset = get_gsm8k_questions(
        model_name="google/gemma-3-1b-it",
        max_prompt_length=256,
        max_completion_length=1024,
        split="train",
    )
    if train_dataset:
        logger.info(f"data_loader :: Loaded {len(train_dataset)} training examples.")
    else:
        logger.error("data_loader :: Failed to load the training dataset.")
