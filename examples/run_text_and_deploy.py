import lightning_tune as lt
from pathlib import Path

config = lt.PipelineConfig(
    model=lt.ModelConfig(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    data=lt.DataConfig(file_path=Path("examples/sample_data_text.csv")),
    trainer=lt.TrainerConfig(max_epochs=1, devices=1),
)
trained_adapter_path = lt.run_finetuning(config)
lt.launch_server(config, trained_adapter_path)
