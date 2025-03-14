import re
import logging

logger = logging.getLogger(__name__)


def extract_xml_answer(text: str) -> str:
    """Extracts the answer from XML-formatted text."""
    logger.debug("utils.extract_xml_answer :: Entering function")
    try:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        answer = answer.strip()
        logger.debug(f"utils.extract_xml_answer :: Extracted answer: {answer}")
        logger.debug("utils.extract_xml_answer :: Exiting function successfully")
        return answer
    except IndexError:
        logger.warning(
            "utils.extract_xml_answer :: IndexError occurred, returning empty string"
        )
        logger.debug("utils.extract_xml_answer :: Exiting function with IndexError")
        return ""  # Or handle the error as appropriate
    except Exception as e:
        logger.error(f"utils.extract_xml_answer :: An unexpected error occurred: {e}")
        logger.debug("utils.extract_xml_answer :: Exiting function with error")
        return ""


def count_xml(text: str) -> float:
    logger.debug("utils.count_xml :: Entering function")
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
        logger.debug("utils.count_xml :: Found '<reasoning>\\n'")
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
        logger.debug("utils.count_xml :: Found '\\n</reasoning>\\n'")
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        logger.debug("utils.count_xml :: Found '\\n<answer>\\n'")
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        logger.debug("utils.count_xml :: Found '\\n</answer>'")
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    logger.debug(f"utils.count_xml :: Count: {count}")
    logger.debug("utils.count_xml :: Exiting function")
    return count
