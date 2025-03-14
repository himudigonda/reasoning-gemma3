from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model
import torch
import logging
from accelerate import dispatch_model

logger = logging.getLogger(__name__)


def load_model_and_processor(
    model_name, lora_r, lora_alpha, lora_target_modules, torch_dtype=torch.float32
):
    """Loads the Gemma-3 model and sets up LoRA for Apple Silicon with memory optimizations."""
    logger.debug(
        f"model.load_model_and_processor :: Entering function with model_name: {model_name}, lora_r: {lora_r}, lora_alpha: {lora_alpha}, lora_target_modules: {lora_target_modules}"
    )

    try:
        logger.info(f"model.load_model_and_processor :: Loading model: {model_name}")

        # Load model config and ensure vocab_size is set
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if not hasattr(config, "vocab_size"):
            config.vocab_size = 32000  # Default vocab size for LLaMA/Gemma models
            logger.warning(
                "model.load_model_and_processor :: vocab_size missing from config, setting default (32000)"
            )

        device = "mps" if torch.backends.mps.is_available() else "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            device_map="auto" if device == "mps" else "cpu",
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        # Enable gradient checkpointing to reduce memory usage
        model.gradient_checkpointing_enable()

        # Offload unused layers to CPU to prevent excessive memory allocation
        model = dispatch_model(model, device_map={"": device})

        logger.info("model.load_model_and_processor :: Model loaded successfully")
    except Exception as e:
        logger.error(f"model.load_model_and_processor :: Failed to load the model: {e}")
        logger.debug("model.load_model_and_processor :: Exiting function with error")
        raise

    try:
        logger.info(
            f"model.load_model_and_processor :: Loading tokenizer: {model_name}"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token_id = 0  # Set padding token ID
        tokenizer.bos_token_id = 1  # Set bos token ID
        tokenizer.eos_token_id = 2  # Set eos token ID
        logger.info("model.load_model_and_processor :: Tokenizer loaded successfully")
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
