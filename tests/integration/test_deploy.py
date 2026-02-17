
import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add src to path
if str(Path(__file__).parents[2] / "src") not in sys.path:
    sys.path.append(str(Path(__file__).parents[2] / "src"))

from lightning_tune.deploy import launch_server
from lightning_tune.config import PipelineConfig, DeploymentConfig, ModelConfig, DataConfig, AudioConfig

class TestDeploy(unittest.TestCase):

    @patch("lightning_tune.deploy.VLLM_AVAILABLE", True)
    @patch("lightning_tune.deploy.LitServer")
    @patch("lightning_tune.deploy.VLLMAPI")
    def test_launch_server_checks_vllm(self, mock_vllm_api, mock_lit_server):
        """Test that VLLM is selected when available and configured."""
        config = PipelineConfig(
            model=ModelConfig(repo_id="dummy"),
            data=DataConfig(file_path=Path("dummy.csv")),
            deployment=DeploymentConfig(use_vllm=True)
        )
        
        launch_server(config, Path("dummy_adapter"))
        
        # Verify VLLMAPI was initialized
        mock_vllm_api.assert_called_once()
        # Verify LitServer was initialized with VLLMAPI instance
        mock_lit_server.assert_called_once()
        args, _ = mock_lit_server.call_args
        self.assertIsInstance(args[0], MagicMock) # The API instance (mocked VLLMAPI return value)

    @patch("lightning_tune.deploy.VLLM_AVAILABLE", True)
    @patch("lightning_tune.deploy.LitServer")
    @patch("lightning_tune.deploy.TextLLMAPI")
    def test_launch_server_force_no_vllm(self, mock_text_api, mock_lit_server):
        """Test that TextLLMAPI is selected when use_vllm=False even if VLLM available."""
        config = PipelineConfig(
            model=ModelConfig(repo_id="dummy"),
            data=DataConfig(file_path=Path("dummy.csv")),
            deployment=DeploymentConfig(use_vllm=False)
        )
        
        launch_server(config, Path("dummy_adapter"))
        
        mock_text_api.assert_called_once()

    @patch("lightning_tune.deploy.LitServer")
    @patch("lightning_tune.deploy.MultiModalAPI")
    def test_launch_server_multimodal(self, mock_mm_api, mock_lit_server):
        """Test that MultiModalAPI is selected for multimodal config."""
        config = PipelineConfig(
            model=ModelConfig(repo_id="dummy"),
            data=DataConfig(
                file_path=Path("dummy.csv"),
                audio_config=AudioConfig(audio_column="a", model_name="d", projection_dim=10)
            )
        )
        self.assertTrue(config.is_multimodal)
        
        launch_server(config, Path("dummy_adapter"))
        
        mock_mm_api.assert_called_once()

if __name__ == "__main__":
    unittest.main()
