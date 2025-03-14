from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch
import logging

logger = logging.getLogger(__name__)


def load_model_and_processor(
    model_name, lora_r, lora_alpha, lora_target_modules, torch_dtype=torch.bfloat16
):
    """Loads the Gemma-3 model and sets up LoRA."""
    logger.debug(
        f"model.load_model_and_processor :: Entering function with model_name: {model_name}, lora_r: {lora_r}, lora_alpha: {lora_alpha}, lora_target_modules: {lora_target_modules}"
    )
    try:
        logger.info(f"model.load_model_and_processor :: Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch_dtype,
            attn_implementation="eager",
        )
        logger.info("model.load_model_and_processor :: Model loaded successfully")
    except Exception as e:
        logger.error(f"model.load_model_and_processor :: Failed to load the model: {e}")
        logger.debug("model.load_model_and_processor :: Exiting function with error")
        raise

    try:
        logger.info(
            f"model.load_model_and_processor :: Loading tokenizer: {model_name}"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = 0  # Set padding token ID
        tokenizer.bos_token_id = 1  # Set bos token ID
        tokenizer.eos_token_id = 2  # Set eos token ID
        logger.info("model.load_model_and_processor :: Tokenizer loaded successfully")
        logger.debug(
            f"model.load_model_and_processor :: pad_token_id: {tokenizer.pad_token_id}, bos_token_id: {tokenizer.bos_token_id}, eos_token_id: {tokenizer.eos_token_id}"
        )
    except Exception as e:
        logger.error(
            f"model.load_model_and_processor :: Failed to load the tokenizer: {e}"
        )
        logger.debug("model.load_model_and_processor :: Exiting function with error")
        raise

    try:
        logger.info("model.load_model_and_processor :: Setting up LoRA")
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info("model.load_model_and_processor :: LoRA setup successfully")
    except Exception as e:
        logger.error(f"model.load_model_and_processor :: Failed to set up LoRA: {e}")
        logger.debug("model.load_model_and_processor :: Exiting function with error")
        raise

    logger.debug("model.load_model_and_processor :: Returning model and tokenizer")
    logger.debug("model.load_model_and_processor :: Exiting function successfully")
    return model, tokenizer
