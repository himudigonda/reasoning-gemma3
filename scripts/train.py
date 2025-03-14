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
    logger.debug("train.main :: Entering function")
    logger.info(f"train.main :: Using configuration file: {config_path}")

    try:
        with open(config_path, "r") as f:
            training_config = yaml.safe_load(f)
        logger.debug("train.main :: Configuration file loaded successfully")
        logger.debug(f"train.main :: Training configuration: {training_config}")
    except FileNotFoundError:
        logger.error(f"train.main :: Configuration file not found: {config_path}")
        logger.debug("train.main :: Exiting function with FileNotFoundError")
        return
    except yaml.YAMLError as e:
        logger.error(f"train.main :: Error parsing YAML file: {e}")
        logger.debug("train.main :: Exiting function with yaml.YAMLError")
        return
    except Exception as e:
        logger.error(
            f"train.main :: An unexpected error occurred while loading the configuration file: {e}"
        )
        logger.debug("train.main :: Exiting function with unexpected error")
        return

    try:
        logger.info("train.main :: Loading training and eval datasets")
        train_dataset = get_gsm8k_questions("train")
        if not train_dataset:
            logger.error(
                "train.main :: Training dataset is empty or could not be loaded."
            )
            logger.debug("train.main :: Exiting function, training dataset empty")
            return
        eval_dataset = get_gsm8k_questions("test")
        if not eval_dataset:
            logger.error("train.main :: Eval dataset is empty or could not be loaded.")
            logger.debug("train.main :: Exiting function, eval dataset empty")
            return
        logger.info("train.main :: Datasets loaded successfully")
        logger.debug(
            f"train.main :: Training dataset size: {len(train_dataset)}, Eval dataset size: {len(eval_dataset)}"
        )
    except Exception as e:
        logger.error(f"train.main :: Failed to load dataset: {e}")
        logger.debug("train.main :: Exiting function with dataset load error")
        return

    try:
        logger.info("train.main :: Loading model and processor")
        model, tokenizer = load_model_and_processor(
            training_config["model_name"],
            training_config["lora_r"],
            training_config["lora_alpha"],
            training_config["lora_target_modules"],
        )
        logger.info("train.main :: Model and processor loaded successfully")
    except Exception as e:
        logger.error(f"train.main :: Failed to load model and processor: {e}")
        logger.debug("train.main :: Exiting function with model load error")
        return

    reward_functions = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ]
    logger.debug(f"train.main :: Reward functions: {reward_functions}")

    try:
        logger.info("train.main :: Creating trainer")
        trainer = create_grpo_trainer(
            model,
            reward_functions,
            training_config,
            train_dataset,
            tokenizer,  # include tokenizer argument
        )

        logger.info("train.main :: Trainer created successfully")
    except Exception as e:
        logger.error(f"train.main :: Failed to create trainer: {e}")
        logger.debug("train.main :: Exiting function with trainer creation error")
        return

    try:
        logger.info("train.main :: Starting training")
        trainer.train()
        logger.info("train.main :: Training completed successfully")
    except Exception as e:
        logger.error(f"train.main :: Training failed: {e}")
        logger.debug("train.main :: Exiting function with training error")
        return

    try:
        logger.info(f"train.main :: Saving model to {training_config['output_dir']}")
        model.save_pretrained(training_config["output_dir"])
        tokenizer.save_pretrained(
            training_config["output_dir"]
        )  # Save tokenizer too # not in line 364, removing from here as well
        logger.info(f"train.main :: Model saved to {training_config['output_dir']}")
    except Exception as e:
        logger.error(f"train.main :: Failed to save model: {e}")
        logger.debug("train.main :: Exiting function with model save error")
        return

    logger.debug("train.main :: Exiting function successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Gemma-3 model with GRPO.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to the training configuration file.",
    )
    args = parser.parse_args()
    main(args.config)
