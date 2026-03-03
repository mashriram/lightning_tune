
import unittest
from unittest.mock import MagicMock, patch
import polars as pl
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parents[2] / "src"))
# Fix potential double stacking
if str(Path(__file__).parents[2] / "src") not in sys.path:
    sys.path.append(str(Path(__file__).parents[2] / "src"))

from lightning_tune.config import PipelineConfig, DataConfig, ModelConfig

class TestConfig(unittest.TestCase):

    def test_data_config_validation(self):
        with self.assertRaises(ValueError):
            DataConfig()
            
        try:
            DataConfig(file_path=Path("data.csv"))
        except ValueError:
            self.fail("DataConfig raised ValueError unexpectedly with file_path")

        try:
            DataConfig(dataset_repo_id="dummy/repo")
        except ValueError:
            self.fail("DataConfig raised ValueError unexpectedly with dataset_repo_id")

    def test_analyze_schema_text(self):
        """Test schema analysis for simple text dataset."""
        # Make unique values high enough or strings long enough to be text not categorical
        df = pl.DataFrame({
            "instruction": [f"Do this {i}" for i in range(100)],
            "input": [f"input {i}" for i in range(100)],
            "output": [f"output {i}" for i in range(100)],
            "id": list(range(100))
        })
        analysis = PipelineConfig._analyze_schema(df)
        self.assertEqual(analysis["output_column"], "output")
        self.assertIn("instruction", analysis["text_columns"])
        self.assertIn("input", analysis["text_columns"])
        self.assertIn("id", analysis["numerical_columns"])
        self.assertIsNone(analysis["image_column"])
        self.assertIsNone(analysis["audio_column"])

    def test_analyze_schema_image(self):
        """Test detection of image columns."""
        df = pl.DataFrame({
            "text": ["caption1", "caption2"],
            "image_path": ["img1.jpg", "img2.jpg"]
        })
        analysis = PipelineConfig._analyze_schema(df)
        self.assertEqual(analysis["image_column"], "image_path")
        self.assertEqual(analysis["output_column"], "text")

    def test_analyze_schema_audio(self):
        """Test detection of audio columns."""
        df = pl.DataFrame({
            "transcript": ["hello", "world"],
            "audio_file": ["audio1.wav", "audio2.mp3"]
        })
        analysis = PipelineConfig._analyze_schema(df)
        self.assertEqual(analysis["audio_column"], "audio_file")
        self.assertEqual(analysis["output_column"], "transcript")

    @patch("lightning_tune.config.PipelineConfig._read_and_sample_dataset")
    @patch("transformers.AutoConfig.from_pretrained")
    def test_from_dataset_local(self, mock_hf_config, mock_read_sample):
        """Test from_dataset with local file."""
        # Mock dataframe
        mock_df = pl.DataFrame({
            "instruction": ["inst"],
            "input": ["inp"],
            "output": ["out"]
        })
        mock_read_sample.return_value = (mock_df, mock_df)
        
        # Mock HF config
        mock_hf_config.return_value.hidden_size = 768
        
        config = PipelineConfig.from_dataset(
            model_repo_id="dummy/model",
            file_path=Path("dummy.csv")
        )
        
        self.assertIsInstance(config, PipelineConfig)
        self.assertEqual(config.data.output_column, "output")
        # Should detect alpaca format
        self.assertEqual(config.data.instruction_column, "instruction")

    @patch("lightning_tune.config.get_dataset_configs")
    @patch("lightning_tune.config.get_dataset_splits")
    @patch("datasets.load_dataset")
    @patch("transformers.AutoConfig.from_pretrained")
    def test_from_dataset_hf(self, mock_hf_config, mock_load_dataset, mock_get_splits, mock_get_configs):
        """Test from_dataset with HF repo."""
        # Mock splits
        mock_get_configs.return_value = ["default"]
        mock_get_splits.return_value = ["train"]
        
        # Mock dataset stream
        mock_ds = [{"instruction": "inst", "output": "out"}]
        mock_load_dataset.return_value = mock_ds
        
        # Mock HF config
        mock_hf_config.return_value.hidden_size = 768
        
        config = PipelineConfig.from_dataset(
            model_repo_id="dummy/model",
            dataset_repo_id="dummy/dataset"
        )
        
        self.assertIsInstance(config, PipelineConfig)
        self.assertEqual(config.data.datasets[0].repo_id, "dummy/dataset")
        self.assertEqual(config.data.datasets[0].split, "train")

if __name__ == "__main__":
    unittest.main()
