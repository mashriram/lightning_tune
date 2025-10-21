from lightning_tune.config import PipelineConfig
from pathlib import Path


def test_smart_config_creation():
    assets_path = Path("tests/assets")
    config = PipelineConfig.from_dataset(
        model_repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        file_path=assets_path / "test_multimodal.csv",
        image_root_path=assets_path,
    )
    assert config.is_multimodal
    assert config.data.vision_config is not None
    assert config.data.tabular_config is not None
    assert config.data.vision_config.image_column == "image_path"
    assert "price" in config.data.tabular_config.numerical_columns
    assert "rating" in config.data.tabular_config.numerical_columns
    assert "category" in config.data.tabular_config.categorical_columns
    assert "description" in config.data.text_columns
    assert config.data.output_column == "review"
