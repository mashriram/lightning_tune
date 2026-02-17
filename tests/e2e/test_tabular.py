import unittest
import shutil
from pathlib import Path
import torch
import polars as pl
import sys

# Add src to path
if str(Path(__file__).parents[2] / "src") not in sys.path:
    sys.path.append(str(Path(__file__).parents[2] / "src"))

from lightning_tune.train import run_finetuning
from lightning_tune.config import PipelineConfig, ModelConfig, DataConfig, TrainConfig, TabularConfig

class TestTabularPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path("temp_tabular_test")
        self.temp_dir.mkdir(exist_ok=True)
        self.data_path = self.temp_dir / "data.csv"
        
        # Tiny model
        self.tiny_llm = "HuggingFaceTB/SmolLM2-135M-Instruct"

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        # Clean up finetuning_runs if needed, but let's keep them for inspection in local dev
        pass

    def test_multimodal_tabular(self):
        """Real text+tabular finetuning."""
        with open(self.data_path, "w") as f:
            f.write("instruction,input,output,price,rarity\n")
            for i in range(10):
                f.write(f"inst{i},inp{i},out{i},{i*10.5},common\n")
        
        config = PipelineConfig(
            model=ModelConfig(repo_id=self.tiny_llm),
            data=DataConfig(
                file_path=self.data_path,
                instruction_column="instruction",
                input_column="input",
                output_column="output",
                tabular_config=TabularConfig(
                    numerical_columns=["price"],
                    categorical_columns=["rarity"],
                    projection_dim=576 # SmolLM hidden size
                )
            ),
            train=TrainConfig(
                batch_size=2,
                llm_lr=1e-4,
                tower_lr=1e-3
            )
        )
        config.trainer.limit_train_batches = 2
        config.trainer.limit_val_batches = 0
        config.trainer.max_epochs = 1
        
        output_path = run_finetuning(config)
        self.assertTrue(output_path.exists())
        self.assertTrue((output_path / "towers.pt").exists()) # Verify towers saved
        self.assertTrue((output_path / "adapter_model.safetensors").exists()) # Verify adapter saved (HF format)

if __name__ == "__main__":
    unittest.main()
