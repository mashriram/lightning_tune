import lightning_tune as lt
from pathlib import Path
import pytest
from PIL import Image
import numpy as np
import json

@pytest.mark.slow
def test_vision_encoder_selection(tmp_path):
    # Create a dummy image file
    image = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    dummy_image_path = image_dir / "test.png"
    image.save(dummy_image_path)

    # Create a dummy alpaca-formatted json file that points to the image
    alpaca_data = f"""
    [
        {{
            "instruction": "Describe this image.",
            "input": "",
            "output": "This is a random noise image.",
            "image": "{dummy_image_path.name}"
        }},
        {{
            "instruction": "What is in this picture?",
            "input": "",
            "output": "Random patterns.",
            "image": "{dummy_image_path.name}"
        }}
    ]
    """
    data_path = tmp_path / "alpaca.json"
    data_path.write_text(alpaca_data)

    # Create a config file
    config_dict = {
        "model": {
            "repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        },
        "data": {
            "file_path": str(data_path),
            "image_root_path": str(image_dir),
            "vision_config": {
                "image_column": "image",
                "model_name": "vit_small_patch16_224",
                "projection_dim": 2048
            }
        },
        "trainer": {
            "max_epochs": 1,
            "evaluation": {
                "do_eval": False
            }
        }
    }
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f)

    config = lt.PipelineConfig.parse_file(str(config_path))

    # Run the multi-modal pipeline
    lt.run_finetuning(config)
