
import unittest
import shutil
import tempfile
from pathlib import Path
import torch
from PIL import Image
import soundfile as sf
import numpy as np
import sys
import os


# Add src to path
if str(Path(__file__).parents[2] / "src") not in sys.path:
    sys.path.append(str(Path(__file__).parents[2] / "src"))

import lightning_tune
print(f"DEBUG: lightning_tune location: {lightning_tune.__file__}")

import torchaudio
try:
    torchaudio.set_audio_backend("soundfile")
except:
    pass

from lightning_tune.train import run_finetuning
from lightning_tune.config import PipelineConfig, ModelConfig, DataConfig, TrainConfig, AudioConfig, VisionConfig

class TestRealE2EPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.data_path = self.test_dir / "data.csv"
        
        # Tiny real models
        self.tiny_llm = "HuggingFaceTB/SmolLM2-135M-Instruct" 
        self.tiny_vision = "vit_tiny_patch16_224" # Valid timm model
        self.tiny_audio = "facebook/wav2vec2-base-960h" # 360MB
        
        # Create dummy assets
        self.img_path = self.test_dir / "image.jpg"
        Image.new('RGB', (224, 224), color='red').save(self.img_path)
        
        self.audio_path = self.test_dir / "audio.wav"
        sf.write(self.audio_path, np.random.uniform(-1, 1, 16000), 16000)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_text_only_finetuning(self):
        """Real text-only finetuning on a tiny model."""
        # Create CSV
        with open(self.data_path, "w") as f:
            f.write("instruction,input,output\n")
            for i in range(5):
                f.write(f"inst{i},inp{i},out{i}\n")
        
        config = PipelineConfig(
            model=ModelConfig(repo_id=self.tiny_llm),
            data=DataConfig(file_path=self.data_path),
            train=TrainConfig(batch_size=1, epochs=1, limit_train_batches=2, limit_val_batches=0)
        )
        # Override trainer kwargs for speed
        config.trainer.limit_train_batches = 2
        config.trainer.limit_val_batches = 0
        config.trainer.max_epochs = 1
        config.trainer.devices = 1
        config.trainer.accelerator = "auto"

        result = run_finetuning(config)
        output_path = result["path"]
        
        self.assertTrue(output_path.exists())
        self.assertTrue((output_path / "adapter_model.safetensors").exists())

    def test_multimodal_vision(self):
        """Real text+vision finetuning."""
        with open(self.data_path, "w") as f:
            f.write("instruction,input,output,image\n")
            for i in range(5):
                f.write(f"inst{i},inp{i},out{i},{self.img_path}\n")

        config = PipelineConfig(
            model=ModelConfig(repo_id=self.tiny_llm),
            data=DataConfig(
                file_path=self.data_path, 
                vision_config=VisionConfig(
                    image_column="image", 
                    model_name=self.tiny_vision,
                    projection_dim=576 # SmolLM hidden size
                ),
                audio_config=None
            )
        )

        config.trainer.limit_train_batches = 2
        config.trainer.limit_val_batches = 0
        config.trainer.max_epochs = 1
        
        result = run_finetuning(config)
        output_path = result["path"]
        self.assertTrue(output_path.exists())

    def test_multimodal_audio(self):
        """Real text+audio finetuning."""
        with open(self.data_path, "w") as f:
            f.write("instruction,input,output,audio\n")
            for i in range(5):
                f.write(f"inst{i},inp{i},out{i},{self.audio_path}\n")

        config = PipelineConfig(
            model=ModelConfig(repo_id=self.tiny_llm),
            data=DataConfig(
                file_path=self.data_path,
                audio_config=AudioConfig(
                    audio_column="audio",
                    model_name=self.tiny_audio,
                    projection_dim=576 # SmolLM2-135M hidden size
                )
            )
        )
        
        config.trainer.limit_train_batches = 2
        config.trainer.limit_val_batches = 0
        config.trainer.max_epochs = 1
        
        result = run_finetuning(config)
        output_path = result["path"]
        self.assertTrue(output_path.exists())

if __name__ == "__main__":
    unittest.main()
