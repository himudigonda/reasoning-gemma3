from trl import GRPOConfig, GRPOTrainer
import torch
import logging

logger = logging.getLogger(__name__)


def create_grpo_trainer(model, tokenizer, reward_funcs, training_config, train_dataset):
    """Creates and configures the GRPOTrainer."""

    try:
        training_args = GRPOConfig(
            learning_rate=training_config["learning_rate"],
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=training_config["weight_decay"],
            warmup_ratio=training_config["warmup_ratio"],
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
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
    except KeyError as e:
        logger.error(f"Missing configuration parameter: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to create GRPOConfig: {e}")
        raise

    try:
        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=train_dataset,
        )
    except Exception as e:
        logger.error(f"Failed to create GRPOTrainer: {e}")
        raise

    return trainer
