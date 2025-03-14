import yaml
import logging
import argparse
from src.data_loader import get_gsm8k_questions
from src.model import load_model_and_processor
from src.reward import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
)
from src.trainer import create_grpo_trainer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main(config_path):
    """Main function to run the training."""
    try:
        with open(config_path, "r") as f:
            training_config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        return
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while loading the configuration file: {e}"
        )
        return

    try:
        train_dataset = get_gsm8k_questions("train")
        if not train_dataset:
            logger.error("Training dataset is empty or could not be loaded.")
            return
        eval_dataset = get_gsm8k_questions("test")
        if not eval_dataset:
            logger.error("Eval dataset is empty or could not be loaded.")
            return
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    try:
        model, tokenizer = load_model_and_processor(
            training_config["model_name"],
            training_config["lora_r"],
            training_config["lora_alpha"],
            training_config["lora_target_modules"],
        )
    except Exception as e:
        logger.error(f"Failed to load model and processor: {e}")
        return

    reward_functions = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ]

    try:
        trainer = create_grpo_trainer(
            model, tokenizer, reward_functions, training_config, train_dataset
        )
    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        return

    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return

    try:
        model.save_pretrained(training_config["output_dir"])
        tokenizer.save_pretrained(training_config["output_dir"])  # Save tokenizer too
        logger.info(
            f"Training complete. Model saved to {training_config['output_dir']}"
        )
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Gemma-3 model with GRPO.")
    parser.add_argument(
        "--config",
        type=str,
        default="gemma3-reasoning/config/training_config.yaml",
        help="Path to the training configuration file.",
    )
    args = parser.parse_args()
    main(args.config)
