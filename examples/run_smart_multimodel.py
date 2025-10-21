import lightning_tune as lt
from pathlib import Path

config = lt.PipelineConfig.from_dataset(
    model_repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    file_path=Path("examples/sample_data_multimodal.csv"),
    image_root_path=Path("examples/"),
)
config.trainer.max_epochs = 2
config.train.batch_size = 1
trained_path = lt.run_finetuning(config)
if trained_path.exists():
    lt.launch_server(config, trained_path)
