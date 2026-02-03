import unittest
from unittest.mock import MagicMock, patch, ANY
import torch
import torch.nn as nn
from src.lightning_tune.model import MultimodalLLM, TabularTower
from src.lightning_tune.config import PipelineConfig, ModelConfig, DataConfig, TrainConfig, PeftConfig, TabularConfig

class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.p1 = nn.Parameter(torch.tensor(1.0))
        self.p1.requires_grad = True
        self.p2 = nn.Parameter(torch.tensor(1.0))
        self.p2.requires_grad = False
        self.dtype = torch.float32

    def forward(self, input_ids=None, inputs_embeds=None, pixel_values=None, labels=None):
        # Determine output based on inputs
        logits = torch.randn(1, 10, 100) # [B, Seq, Vocab]
        return MagicMock(logits=logits, loss=torch.tensor(0.5))

    def get_input_embeddings(self):
        return nn.Embedding(100, 32) # vocab 100, dim 32

    def print_trainable_parameters(self):
        pass

class TestNativeVLMWithTabular(unittest.TestCase):
    def setUp(self):
        # Properly initialize config with cardinality dict
        self.config = PipelineConfig(
            model=ModelConfig(repo_id="test/native-vlm"),
            data=DataConfig(
                file_path=None, dataset_repo_id="test/ds", text_columns=["text"], output_column="out",
                tabular_config=TabularConfig(
                    numerical_columns=["num"],
                    categorical_columns=[],
                    projection_dim=32,
                    categorical_cardinality={} # Important
                )
            ),
            train=TrainConfig(peft=PeftConfig(r=4))
        )

    @patch("src.lightning_tune.model.AutoConfig.from_pretrained")
    @patch("src.lightning_tune.model.AutoModelForVision2Seq.from_pretrained")
    @patch("src.lightning_tune.model.get_peft_model")
    def test_native_vlm_tabular_forward(self, mock_get_peft, mock_v2s, mock_config):
        # Setup mocks
        mock_hf_config = MagicMock()
        mock_hf_config.vision_config = {"exists": True}
        mock_config.return_value = mock_hf_config

        fake_llm = FakeModel()
        mock_v2s.return_value = fake_llm
        mock_get_peft.return_value = fake_llm

        model = MultimodalLLM(self.config)

        # Ensure tabular tower exists
        self.assertIsNotNone(model.tabular_tower)

        # Test Forward
        batch = {
            "input_ids": torch.randint(0, 100, (1, 10)),
            "labels": torch.randint(0, 100, (1, 10)),
            "image": torch.randn(1, 3, 224, 224),
            "tabular_num": torch.randn(1, 1) # 1 numerical column
        }

        # We need to spy on the LLM forward call to verify inputs_embeds was passed
        with patch.object(fake_llm, "forward", side_effect=fake_llm.forward) as spy_forward:
            logits, prefix = model(batch)

            # Verify forward called
            # It might be called twice if label shifting occurs (which we implemented)
            # The second call is the one we care about most as it has inputs_embeds

            # Find the call with inputs_embeds
            found_embeds_call = False
            for call in spy_forward.call_args_list:
                kwargs = call[1]
                if kwargs.get("inputs_embeds") is not None:
                    found_embeds_call = True
                    self.assertIsNone(kwargs.get("input_ids"))
                    self.assertIsNotNone(kwargs.get("pixel_values"))

                    # Check inputs_embeds shape
                    # Tabular adds 1 token (projection_dim 32)
                    # Text is 10 tokens
                    # Total should be 11
                    expected_shape = (1, 11, 32)
                    self.assertEqual(kwargs["inputs_embeds"].shape, expected_shape)

            self.assertTrue(found_embeds_call, "Forward should have been called with inputs_embeds")

if __name__ == "__main__":
    unittest.main()
