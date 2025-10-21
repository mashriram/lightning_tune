# ⚡ Lightning Tune: A Production-Ready Finetuning Library

Lightning Tune is a powerful, production-grade Python library for finetuning and deploying both text-only and multi-modal Large Language Models. Built entirely on the Lightning AI ecosystem (`PyTorch Lightning`, `LitGPT`, `LitServe`), it provides a seamless, automated, and efficient workflow from raw data to a deployed API.

## Features

- **Smart Configuration**: Automatically analyze your dataset's schema and size to determine a strong baseline configuration for finetuning.
- **Smart Hyperparameter Tuning**: Infers sensible starting points for **Learning Rate**, **Number of Epochs**, and **LoRA Rank** based on your dataset size.
- **Flexible Multi-Modal Architecture**: Define multi-modal towers explicitly in your code or let the smart factory detect them for you.
- **Advanced PEFT Methods**: Natively supports `QLoRA`, `LoRA`, and `DoRA` for text-only finetuning via Hugging Face's `SFTTrainer`.
- **End-to-End Workflow**: A unified API takes you from training to a production-ready inference server.
- **Interactive UI**: A Gradio-based web interface for a no-code, point-and-click finetuning experience.

## Installation

```bash
# Clone the repository (or create the project from the provided code)
cd lightning-tune

# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install the library and its testing dependencies in editable mode
uv pip install -e ".[test]"
```

## How to Use

### 1. The Smart Factory (Recommended)

Let the library analyze your dataset and choose the best starting configuration.

```python
# examples/run_smart_multimodal.py
import lightning_tune as lt
from pathlib import Path

# The model repo_id is a mandatory argument
config = lt.PipelineConfig.from_dataset(
    model_repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    file_path=Path("examples/sample_data_multimodal.csv"),
    image_root_path=Path("examples/")
)

# You can still override any smart setting
config.trainer.max_epochs = 2

trained_model_path = lt.run_finetuning(config)
if trained_model_path.exists():
    lt.launch_server(config, trained_model_path)
```

### 2. Manual Configuration (Full Control)

Explicitly define every aspect of your pipeline, including multi-modal towers.

```python
# examples/run_manual_multimodal.py
import lightning_tune as lt
from pathlib import Path

config = lt.PipelineConfig(
    model=lt.ModelConfig(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    data=lt.DataConfig(
        file_path=Path("examples/sample_data_multimodal.csv"),
        image_root_path=Path("examples/"),
        text_columns=["product_description"],
        output_column="customer_review",
        vision_config=lt.VisionConfig(
            image_column="image_path",
            projection_dim=2048 # Manually set to match TinyLlama's hidden size
        ),
        tabular_config=lt.TabularConfig(
            numerical_columns=["price"],
            categorical_columns=["category"],
            projection_dim=2048
        )
    ),
    trainer=lt.TrainerConfig(max_epochs=2),
    train=lt.TrainConfig(batch_size=1)
)
lt.run_finetuning(config)
```

### 3. Using the Gradio Web UI

For a seamless, no-code experience, launch the interactive Gradio application.

```bash
python app/main.py
```

### 4. Running Tests

```bash
pytest
```

