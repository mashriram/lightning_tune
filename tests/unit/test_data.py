
import unittest
from unittest.mock import MagicMock, patch
import torch
import sys
from pathlib import Path

# Add src to path
if str(Path(__file__).parents[2] / "src") not in sys.path:
    sys.path.append(str(Path(__file__).parents[2] / "src"))

from lightning_tune.data import MultiModalDataset, multimodal_collate_fn
from lightning_tune.config import PipelineConfig, DataConfig, AudioConfig, ModelConfig

class TestData(unittest.TestCase):

    @patch("lightning_tune.data.torchaudio.load")
    @patch("lightning_tune.data.pl.read_csv")
    def test_audio_loading(self, mock_read_csv, mock_audio_load):
        """Test audio loading and resampling."""
        # Mock dataframe
        mock_df = MagicMock()
        mock_df.__len__.return_value = 1
        mock_df.row.return_value = {"text": "hello", "audio": "path/to/audio.wav", "output": "world"}
        mock_read_csv.return_value = mock_df

        # Mock audio load (simulating 48kHz stereo)
        mock_audio_load.return_value = (torch.randn(2, 48000), 48000)

        # Config
        config = PipelineConfig(
            model=ModelConfig(repo_id="dummy/model"),
            data=DataConfig(
                file_path=Path("dummy.csv"),
                audio_config=AudioConfig(audio_column="audio", projection_dim=768, model_name="dummy"),
                text_columns=["text"],
                output_column="output"
            )
        )
        
        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]

        dataset = MultiModalDataset(config, tokenizer)
        
        # Start patch for Resample since it's instantiated in __init__?
        # Actually Resample is used in _load_audio if rate mismatch.
        # But also in __init__ _setup_audio_transforms
        
        with patch("lightning_tune.data.Resample") as mock_resample_cls:
            mock_resample_instance = MagicMock()
            mock_resample_instance.side_effect = lambda x: x[:, ::3] # rough downsample simulation
            mock_resample_cls.return_value = mock_resample_instance
            
            item = dataset[0]
            
            # Verify audio is present
            self.assertIn("audio", item)
            # Verify shape (mono after mean(0))
            # Input was stereo (2, 48000). Resampled to (2, 16000) roughly. Mean -> (16000)
            self.assertEqual(item["audio"].dim(), 1)

    def test_collate_fn(self):
        """Test multimodal collate function padding."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        
        # Batch with audio of different lengths
        batch = [
            {
                "input_ids": torch.tensor([1, 2]),
                "labels": torch.tensor([1, 2]),
                "audio": torch.randn(16000) # 1s
            },
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([1, 2, 3]),
                "audio": torch.randn(32000) # 2s
            }
        ]
        
        collated = multimodal_collate_fn(batch, tokenizer)
        
        self.assertIn("audio", collated)
        self.assertEqual(collated["audio"].shape[0], 2)
        # Should align to max length (32000)
        self.assertEqual(collated["audio"].shape[1], 32000)
        # First item should be padded
        # Check padding (assuming padding is at end? F.pad defaults to end)
        self.assertTrue(torch.all(collated["audio"][0, 16000:] == 0))

if __name__ == "__main__":
    unittest.main()
