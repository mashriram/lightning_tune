
import unittest
from unittest.mock import MagicMock, patch
import torch
import sys
from pathlib import Path

# Add src to path
if str(Path(__file__).parents[2] / "src") not in sys.path:
    sys.path.append(str(Path(__file__).parents[2] / "src"))

from lightning_tune.model import AudioTower, MultimodalLLM
from lightning_tune.config import PipelineConfig, DataConfig, AudioConfig, ModelConfig

class TestModel(unittest.TestCase):

    def test_audio_tower_forward(self):
        """Test AudioTower forward pass."""
        config = AudioConfig(audio_column="audio", model_name="dummy/audio", projection_dim=768)
        
        with patch("transformers.AutoModel.from_pretrained") as mock_model:
            mock_backbone = MagicMock()
            mock_backbone.config.hidden_size = 768
            # Output with last_hidden_state [B, T, D]
            output = MagicMock()
            output.last_hidden_state = torch.randn(2, 50, 768)
            mock_backbone.return_value = output
            mock_model.return_value = mock_backbone
            
            tower = AudioTower(config)
            
            # Input [B, T]
            dummy_input = torch.randn(2, 16000)
            out = tower(dummy_input)
            
            # Adjusted: AudioTower in code returns [B, 1, proj_dim] after linear projection
            self.assertEqual(out.shape, (2, 1, 768))

    def test_multimodal_llm_forward_with_audio(self):
        """Test MultimodalLLM forward pass adding audio embeddings."""
        # Config
        pipe_config = PipelineConfig(
            model=ModelConfig(repo_id="dummy/llm"),
            data=DataConfig(
                dataset_repo_id="dummy/ds",
                audio_config=AudioConfig(audio_column="audio", model_name="dummy/audio", projection_dim=768)
            )
        )
        
        with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_llm_cls, \
             patch("transformers.AutoModel.from_pretrained") as mock_audio_cls, \
             patch("transformers.AutoTokenizer.from_pretrained"), \
             patch("transformers.AutoConfig.from_pretrained") as mock_config, \
             patch("lightning_tune.model.get_peft_model") as mock_peft:
            
            # Mock Config
            mock_cfg_instance = MagicMock()
            mock_cfg_instance.vision_config = None
            mock_cfg_instance.architectures = []
            mock_config.return_value = mock_cfg_instance
            
            # Mock LLM
            mock_llm = MagicMock()
            mock_llm.dtype = torch.float32
            mock_llm.get_input_embeddings.return_value = torch.nn.Embedding(100, 768)
            # Mock forward output
            mock_llm_output = MagicMock()
            mock_llm_output.logits = torch.randn(1, 10, 100) # [B, Seq, Vocab]
            mock_llm.return_value = mock_llm_output
            mock_llm_cls.return_value = mock_llm
            mock_peft.return_value = mock_llm # Pass through

            # Mock Audio Tower backbone
            mock_backbone = MagicMock()
            mock_backbone.config.hidden_size = 768
            mock_backbone_out = MagicMock()
            mock_backbone_out.last_hidden_state = torch.randn(1, 100, 768)
            mock_backbone.return_value = mock_backbone_out
            mock_audio_cls.return_value = mock_backbone
            
            model = MultimodalLLM(pipe_config)
            
            # Batch
            batch = {
                "input_ids": torch.randint(0, 100, (1, 10)),
                "labels": torch.randint(0, 100, (1, 10)),
                "audio": torch.randn(1, 16000)
            }
            
            logits, num_prefix = model(batch)
            
            # LLM is called with inputs_embeds
            # Inputs embeds = [Audio(1), Text(10)] -> Total 11
            # But the mock LLM returns logits based on its own logic (mocked)
            # Check call args
            call_kwargs = mock_llm.call_args.kwargs
            self.assertIn("inputs_embeds", call_kwargs)
            # Shape check: [1, 11, 768] (1 audio + 10 text)
            self.assertEqual(call_kwargs["inputs_embeds"].shape, (1, 11, 768))
            self.assertEqual(num_prefix, 1)

if __name__ == "__main__":
    unittest.main()
