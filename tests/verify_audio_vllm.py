import torch
import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parents[1] / "src"))

from lightning_tune.config import PipelineConfig, AudioConfig, DataConfig, ModelConfig
from lightning_tune.model import AudioTower, MultimodalLLM
from lightning_tune.deploy import VLLMAPI, VLLM_AVAILABLE

class TestAudioAndVLLM(unittest.TestCase):
    def test_audio_tower(self):
        print("\nTesting AudioTower...")
        config = AudioConfig(model_name="facebook/wav2vec2-base-960h", projection_dim=768)
        # Mock AutoModel to avoid downloading weights
        with patch("transformers.AutoModel.from_pretrained") as mock_model:
            mock_backbone = MagicMock()
            mock_backbone.config.hidden_size = 768
            # output of backbone
            mock_output = MagicMock()
            # shape [B, T, D]
            mock_output.last_hidden_state = torch.randn(2, 100, 768)
            mock_backbone.return_value = mock_output
            mock_model.return_value = mock_backbone
            
            tower = AudioTower(config)
            
            # Dummy audio input [B, T]
            dummy_input = torch.randn(2, 16000)
            output = tower(dummy_input)
            
            print(f"AudioTower output shape: {output.shape}")
            # Expected: [B, 1, projection_dim] (unsqueeze(1) is used in forward)
            self.assertEqual(output.shape, (2, 1, 768))

    def test_multimodal_llm_init_audio(self):
        print("\nTesting MultimodalLLM with Audio...")
        # Mock configs
        data_cfg = DataConfig(
            dataset_repo_id="dummy/dataset",
            audio_config=AudioConfig(model_name="dummy/audio", projection_dim=768)
        )
        model_cfg = ModelConfig(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        pipe_cfg = PipelineConfig(model=model_cfg, data=data_cfg)
        
        # Mock AutoModel stuff
        with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_llm, \
             patch("transformers.AutoModel.from_pretrained") as mock_audio, \
             patch("transformers.AutoTokenizer.from_pretrained") as mock_tok, \
             patch("lightning_tune.model.get_peft_model") as mock_peft:
            
            mock_llm_instance = MagicMock()
            mock_llm_instance.dtype = torch.float32
            mock_llm.return_value = mock_llm_instance
            
            mock_audio_instance = MagicMock()
            mock_audio_instance.config.hidden_size = 768
            mock_audio.return_value = mock_audio_instance

            model = MultimodalLLM(pipe_cfg)
            
            self.assertIsNotNone(model.audio_tower)
            print("MultimodalLLM initialized with AudioTower.")

    def test_vllm_api_import_and_init(self):
        print("\nTesting VLLMAPI...")
        if not VLLM_AVAILABLE:
            print("VLLM not installed, skipping runtime test.")
            return

        # If installed, try to instantiate (mocking LLM engine to avoid GPU)
        config = PipelineConfig(model=ModelConfig(repo_id="dummy"))
        adapter_path = Path("dummy/adapter")
        
        with patch("vllm.LLM") as mock_llm_cls:
             api = VLLMAPI(adapter_path, config)
             api.setup("cuda")
             self.assertIsNotNone(api.llm_engine)
             print("VLLMAPI initialized.")

if __name__ == "__main__":
    unittest.main()
