from trl import GRPOConfig, GRPOTrainer
import torch
import logging

logger = logging.getLogger(__name__)


def create_grpo_trainer(model, reward_funcs, training_config, train_dataset):
    """Creates and configures the GRPOTrainer."""
    logger.debug("trainer.create_grpo_trainer :: Entering function")

    try:
        logger.info("trainer.create_grpo_trainer :: Creating GRPOConfig")
        training_args = GRPOConfig(
            learning_rate=training_config["learning_rate"],
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=training_config["weight_decay"],
            warmup_ratio=training_config["warmup_ratio"],
            lr_scheduler_type="cosine",
            optim="adamw_torch",  # Changed from "adamw_8bit" to be compatible with MPS
            logging_steps=training_config["logging_steps"],
            per_device_train_batch_size=training_config["batch_size"],
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            num_generations=training_config["num_generations"],
            max_prompt_length=training_config["max_prompt_length"],
            max_completion_length=training_config["max_completion_length"],
            num_train_epochs=training_config["num_epochs"],
            max_steps=training_config["max_steps"],
            save_steps=training_config["save_steps"],
            max_grad_norm=0.1,
            report_to=training_config["report_to"],
        )
        logger.info("trainer.create_grpo_trainer :: GRPOConfig created successfully")
        logger.debug(
            f"trainer.create_grpo_trainer :: GRPOConfig parameters: {training_args}"
        )
    except KeyError as e:
        logger.error(
            f"trainer.create_grpo_trainer :: Missing configuration parameter: {e}"
        )
        logger.debug("trainer.create_grpo_trainer :: Exiting function with KeyError")
        raise
    except Exception as e:
        logger.error(f"trainer.create_grpo_trainer :: Failed to create GRPOConfig: {e}")
        logger.debug("trainer.create_grpo_trainer :: Exiting function with error")
        raise

    try:
        logger.info("trainer.create_grpo_trainer :: Creating GRPOTrainer")
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_funcs,  # Removed tokenizer from here
            args=training_args,
            train_dataset=train_dataset,
        )
        logger.info("trainer.create_grpo_trainer :: GRPOTrainer created successfully")
    except Exception as e:
        logger.error(
            f"trainer.create_grpo_trainer :: Failed to create GRPOTrainer: {e}"
        )
        logger.debug("trainer.create_grpo_trainer :: Exiting function with error")
        raise

    logger.debug("trainer.create_grpo_trainer :: Returning trainer")
    logger.debug("trainer.create_grpo_trainer :: Exiting function successfully")
    return trainer
