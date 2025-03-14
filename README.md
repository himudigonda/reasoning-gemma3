# Gemma-3 Reasoning Training with GRPO

This project fine-tunes the **Gemma-3** model to enhance its reasoning abilities using the **GRPO (Generative Reward Policy Optimization)** framework. The training process leverages the GSM8k dataset and is designed to run efficiently on GPU hardware.

---

## Table of Contents

- [Gemma-3 Reasoning Training with GRPO](#gemma-3-reasoning-training-with-grpo)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Setup](#setup)
  - [Usage](#usage)
  - [How to Execute the Codebase](#how-to-execute-the-codebase)
  - [CLI Execution and Flags](#cli-execution-and-flags)
  - [Important Considerations](#important-considerations)
  - [Notes](#notes)

---

## Overview

The goal of this project is to improve the reasoning capabilities of the Gemma-3 model by fine-tuning it with the GRPO framework. The GSM8k dataset, a collection of grade-school math problems, is used to train the model. This README provides instructions for setting up the environment, installing dependencies, configuring the training, and running the codebase.

---

## Setup

To get started, follow these steps:

1. **Create a Virtual Environment**  
   Use `conda` or `venv` to create an isolated Python environment. For example, with `conda`:  
   ```bash
   conda create -n gemma3 python=3.10 -y
   conda activate gemma3
   ```

2. **Install Dependencies**  
   Install the required Python packages listed in `requirements.txt`:  
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the GSM8k Dataset**  
   The dataset is automatically downloaded at runtime using the `datasets` library. Ensure you have an active internet connection the first time you run the training script.

4. **Configure Training**  
   Edit the `config/training_config.yaml` file to adjust hyperparameters such as learning rate, batch size, and other training settings.

---

## Usage

To train the model, run the following command:  
```bash
python scripts/train.py --config config/training_config.yaml
```

If you want to use the default configuration file, simply omit the `--config` flag:  
```bash
python scripts/train.py
```

---

## How to Execute the Codebase

Follow these steps to set up and run the project:

1. **Set Up the Environment**  
   ```bash
   conda create -n gemma3 python=3.10 -y
   conda activate gemma3
   ```

2. **Run the Setup Script**  
   Make the setup script executable and run it:  
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the GSM8k Dataset**  
   This step is handled automatically at runtime by the `get_gsm8k_questions` function from the `datasets` library.

5. **Configure the Training**  
   Open `config/training_config.yaml` in a text editor and adjust the hyperparameters as needed.

6. **Run the Training**  
   Use one of the following commands:  
   - With a custom config:  
     ```bash
     python scripts/train.py --config config/training_config.yaml
     ```
   - With the default config:  
     ```bash
     python scripts/train.py
     ```

---

## CLI Execution and Flags

The main training script (`train.py`) supports the following command-line interface:

- **`scripts/train.py --config path/to/config.yaml`**  
  - **Description**: Starts the training process with a specified configuration file.
  - **`--config`**: Path to the training configuration file (e.g., `config/training_config.yaml`). If omitted, the script defaults to the configuration file at `config/training_config.yaml`.

Example:  
```bash
python scripts/train.py --config custom_config.yaml
```

---

## Important Considerations

- **Error Handling**: Basic error handling is implemented using `try...except` blocks and logging. Enhance this further for robustness, especially around file I/O and API calls.
- **Dataset Download**: The GSM8k dataset is downloaded automatically on the first run of `get_gsm8k_questions`. Ensure internet access is available.
- **GPU Availability**: Training requires a compatible GPU with PyTorch installed. Verify your setup before running the script.
- **Hyperparameter Tuning**: The default values in `training_config.yaml` are starting points. Experiment with different settings to optimize performance for your hardware and dataset.
- **Transformers Version**: Ensure the version of the `transformers` library specified in `requirements.txt` is compatible with your setup to avoid version mismatch issues.

---

## Notes

- A GPU is strongly recommended for training to achieve reasonable performance.
- Adjust hyperparameters in `training_config.yaml` based on your specific use case and hardware capabilities.
- The codebase is designed to be flexible—modify the configuration file instead of hardcoding values in the scripts.
