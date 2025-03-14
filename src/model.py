from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch
import logging

logger = logging.getLogger(__name__)


def load_model_and_processor(
    model_name, lora_r, lora_alpha, lora_target_modules, torch_dtype=torch.bfloat16
):
    """Loads the Gemma-3 model and sets up LoRA."""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch_dtype,
            attn_implementation="eager",
        )
    except Exception as e:
        logger.error(f"Failed to load the model: {e}")
        raise

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = 0  # Set padding token ID
        tokenizer.bos_token_id = 1  # Set bos token ID
        tokenizer.eos_token_id = 2  # Set eos token ID
    except Exception as e:
        logger.error(f"Failed to load the tokenizer: {e}")
        raise

    try:
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
    except Exception as e:
        logger.error(f"Failed to set up LoRA: {e}")
        raise

    return model, tokenizer
