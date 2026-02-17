import unittest
from unittest.mock import MagicMock, patch, ANY
import torch
import torch.nn as nn
from src.lightning_tune.model import MultimodalLLM
from src.lightning_tune.config import PipelineConfig, ModelConfig, DataConfig, TrainConfig, PeftConfig

class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.p1 = nn.Parameter(torch.tensor(1.0))
        self.p1.requires_grad = True
        self.p2 = nn.Parameter(torch.tensor(1.0))
        self.p2.requires_grad = False
        self.dtype = torch.float32

    def print_trainable_parameters(self):
        pass

class TestNativeVLM(unittest.TestCase):
    def setUp(self):
        self.config = PipelineConfig(
            model=ModelConfig(repo_id="test/native-vlm"),
            data=DataConfig(file_path=None, dataset_repo_id="test/ds", text_columns=["text"], output_column="out"),
            train=TrainConfig(peft=PeftConfig(r=4))
        )

    @patch("src.lightning_tune.model.AutoConfig.from_pretrained")
    @patch("src.lightning_tune.model.AutoModelForVision2Seq.from_pretrained")
    @patch("src.lightning_tune.model.get_peft_model")
    def test_native_vlm_initialization(self, mock_get_peft, mock_v2s, mock_config):
        # Setup mocks
        mock_hf_config = MagicMock()
        mock_hf_config.vision_config = {"exists": True}
        mock_config.return_value = mock_hf_config

        mock_llm = MagicMock()
        mock_llm.dtype = torch.float32
        mock_llm.parameters.return_value = [MagicMock(requires_grad=False)]
        mock_v2s.return_value = mock_llm

        # Setup PEFT return mock with dtype
        mock_peft = MagicMock()
        mock_peft.dtype = torch.float32
        mock_get_peft.return_value = mock_peft

        # Init model
        model = MultimodalLLM(self.config)

        # Assert native flag is set
        self.assertTrue(model.is_native_vlm)

        # Assert PEFT model was created
        mock_get_peft.assert_called_once()

        # Verify that we tried to load AutoModelForVision2Seq
        mock_v2s.assert_called_once()

    @patch("src.lightning_tune.model.AutoConfig.from_pretrained")
    @patch("src.lightning_tune.model.AutoModelForVision2Seq.from_pretrained")
    @patch("src.lightning_tune.model.get_peft_model")
    def test_configure_optimizers(self, mock_get_peft, mock_v2s, mock_config):
        mock_hf_config = MagicMock()
        mock_hf_config.vision_config = {"exists": True}
        mock_config.return_value = mock_hf_config

        # Use FakeModel for LLM so parameters() works naturally
        fake_llm = FakeModel()
        mock_v2s.return_value = fake_llm

        # Simulating PEFT: returns a model where some params are trainable.
        # Since fake_llm gets frozen in __init__, we need get_peft_model to return
        # something with trainable params.

        # We can use a side_effect for get_peft_model that unfreezes p1
        def peft_side_effect(model, config):
            model.p1.requires_grad = True # Simulating LoRA unfreezing or adding adapters
            return model

        mock_get_peft.side_effect = peft_side_effect

        model = MultimodalLLM(self.config)

        optimizer = model.configure_optimizers()

        self.assertEqual(len(optimizer.param_groups), 1)
        # Check that p1 is in optimizer and p2 is not
        params_in_opt = optimizer.param_groups[0]["params"]
        self.assertTrue(any(p is fake_llm.p1 for p in params_in_opt))
        self.assertFalse(any(p is fake_llm.p2 for p in params_in_opt))

if __name__ == "__main__":
    unittest.main()
