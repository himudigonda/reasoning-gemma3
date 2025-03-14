import re
from src.utils import extract_xml_answer
import logging

logger = logging.getLogger(__name__)


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    logger.debug("reward.correctness_reward_func :: Entering function")
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    logger.debug(
        f"reward.correctness_reward_func :: Question: {q}, Answer: {answer[0]}, Response: {responses[0]}, Extracted: {extracted_responses[0]}"
    )
    rewards = [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
    logger.debug(f"reward.correctness_reward_func :: Rewards: {rewards}")
    logger.debug("reward.correctness_reward_func :: Exiting function")
    return rewards


def int_reward_func(completions, **kwargs) -> list[float]:
    logger.debug("reward.int_reward_func :: Entering function")
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards = [0.5 if r.isdigit() else 0.0 for r in extracted_responses]
    logger.debug(f"reward.int_reward_func :: Rewards: {rewards}")
    logger.debug("reward.int_reward_func :: Exiting function")
    return rewards


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    logger.debug("reward.strict_format_reward_func :: Entering function")
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    rewards = [0.5 if match else 0.0 for match in matches]
    logger.debug(f"reward.strict_format_reward_func :: Rewards: {rewards}")
    logger.debug("reward.strict_format_reward_func :: Exiting function")
    return rewards


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    logger.debug("reward.soft_format_reward_func :: Entering function")
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    rewards = [0.5 if match else 0.0 for match in matches]
    logger.debug(f"reward.soft_format_reward_func :: Rewards: {rewards}")
    logger.debug("reward.soft_format_reward_func :: Exiting function")
    return rewards


from src.utils import count_xml


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    logger.debug("reward.xmlcount_reward_func :: Entering function")
    contents = [completion[0]["content"] for completion in completions]
    rewards = [count_xml(c) for c in contents]
    logger.debug(f"reward.xmlcount_reward_func :: Rewards: {rewards}")
    logger.debug("reward.xmlcount_reward_func :: Exiting function")
    return rewards
