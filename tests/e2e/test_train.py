
import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add src to path
if str(Path(__file__).parents[2] / "src") not in sys.path:
    sys.path.append(str(Path(__file__).parents[2] / "src"))

from lightning_tune.train import run_finetuning
from lightning_tune.config import PipelineConfig, ModelConfig, DataConfig, TrainConfig, PeftConfig, VisionConfig

class TestTrainFlow(unittest.TestCase):

    @patch("lightning_tune.train._run_text_finetuning_pipeline")
    @patch("lightning_tune.train._run_multimodal_pipeline")
    def test_run_finetuning_dispatch(self, mock_mm, mock_text):
        """Test dispatch logic."""
        # Text only
        config_text = PipelineConfig(
            model=ModelConfig(repo_id="dummy"),
            data=DataConfig(file_path=Path("dummy.csv"))
        )
        run_finetuning(config_text)
        mock_text.assert_called_once()
        mock_mm.assert_not_called()
        
        mock_text.reset_mock()
        
        # Multimodal (audio)
        config_mm = PipelineConfig(
            model=ModelConfig(repo_id="dummy"),
            data=DataConfig(
                file_path=Path("dummy.csv"),
                # We need one multimodal config present
                vision_config=VisionConfig(image_column="img", projection_dim=128)
            )
        )
        # Fix hack properly
        config_mm.data.vision_config = MagicMock()
        
        run_finetuning(config_mm)
        mock_mm.assert_called_once()
        mock_text.assert_not_called()

    @patch("lightning_tune.train.SFTTrainer")
    @patch("lightning_tune.train.AutoModelForCausalLM.from_pretrained")
    @patch("lightning_tune.train.AutoTokenizer.from_pretrained")
    @patch("lightning_tune.train.prepare_text_dataset")
    def test_text_finetuning_dora(self, mock_prep_data, mock_tok, mock_model, mock_trainer):
        """Test text finetuning setup with DoRA."""
        config = PipelineConfig(
            model=ModelConfig(repo_id="dummy"),
            data=DataConfig(file_path=Path("dummy.csv")),
            train=TrainConfig(peft=PeftConfig(method="dora"))
        )
        
        # Mock dataset
        mock_prep_data.return_value = {"train": MagicMock()}

        from lightning_tune.train import _run_text_finetuning_pipeline
        _run_text_finetuning_pipeline(config)
        
        # Verify call to SFTTrainer
        mock_trainer.assert_called_once()
        _, kwargs = mock_trainer.call_args
        peft_config = kwargs["peft_config"]
        # Verify use_dora is True
        self.assertTrue(peft_config.use_dora)

    @patch("lightning_tune.train.L.Trainer")
    @patch("lightning_tune.train.MultimodalLLM")
    @patch("lightning_tune.train.DataLoader")
    @patch("lightning_tune.train.MultiModalDataset")
    @patch("lightning_tune.train.AutoTokenizer.from_pretrained")
    @patch("lightning_tune.train.AutoConfig.from_pretrained")
    @patch("lightning_tune.train.torch.save")
    def test_multimodal_pipeline(self, mock_save, mock_cfg, mock_tok, mock_ds, mock_dl, mock_model, mock_trainer):
        """Test multimodal pipeline setup."""
        config = PipelineConfig(
            model=ModelConfig(repo_id="dummy"),
            data=DataConfig(file_path=Path("dummy.csv"), vision_config=VisionConfig(image_column="img", projection_dim=128))
        )
        config.data.vision_config = VisionConfig(image_column="img", projection_dim=128) # Ensure truthy

        from lightning_tune.train import _run_multimodal_pipeline
        _run_multimodal_pipeline(config)
        
        mock_model.assert_called_once()
        mock_trainer.assert_called_once()
        # Verify trainer.fit called
        mock_trainer.return_value.fit.assert_called_once()

if __name__ == "__main__":
    unittest.main()
