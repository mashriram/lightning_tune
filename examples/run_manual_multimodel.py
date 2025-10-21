import lightning_tune as lt
from pathlib import Path

config = lt.PipelineConfig(
    model=lt.ModelConfig(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    data=lt.DataConfig(
        file_path=Path("examples/sample_data_multimodal.csv"),
        image_root_path=Path("examples/"),
        text_columns=["product_description"],
        output_column="customer_review",
        vision_config=lt.VisionConfig(image_column="image_path", projection_dim=2048),
        tabular_config=lt.TabularConfig(
            numerical_columns=["price"],
            categorical_columns=["category"],
            projection_dim=2048,
        ),
    ),
    trainer=lt.TrainerConfig(max_epochs=2),
    train=lt.TrainConfig(batch_size=1),
)
trained_path = lt.run_finetuning(config)
if trained_path.exists():
    lt.launch_server(config, trained_path)
