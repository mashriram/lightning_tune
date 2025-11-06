import pytest
import lightning_tune as lt
from pathlib import Path


@pytest.mark.slow
def test_e2e_text_pipeline(tmp_path):
    (tmp_path / "text.csv").write_text("instruction,input,output\ntest,test,test")
    config = lt.PipelineConfig(
        model=lt.ModelConfig(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        data=lt.DataConfig(file_path=tmp_path / "text.csv"),
    )
    config.trainer.max_epochs = 1
    config.trainer.evaluation.do_eval = False
    output_path = lt.run_finetuning(config)
    assert (output_path / "adapter_model.safetensors").exists()


@pytest.mark.slow
def test_e2e_multimodal_pipeline(tmp_path):
    test_dir = Path(__file__).parent
    config = lt.PipelineConfig.from_dataset(
        model_repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        file_path=test_dir / "assets/test_multimodal.csv",
        image_root_path=test_dir / "assets",
    )
    config.trainer.max_epochs = 1
    config.train.batch_size = 1
    output_path = lt.run_finetuning(config)
    assert output_path.exists() and (output_path.parent / "preprocessors.pt").exists()
