import lightning_tune as lt
from pathlib import Path
from unittest.mock import patch

def test_evaluation(tmp_path):
    # Create a dummy alpaca-formatted json file
    alpaca_data = """
    [
        {
            "instruction": "Give me a recipe for a chocolate cake.",
            "input": "",
            "output": "1. Preheat oven to 350 F. 2. Mix flour, sugar, and cocoa powder. 3. Add eggs, milk, and oil. 4. Bake for 30 minutes."
        },
        {
            "instruction": "Translate the following sentence to French.",
            "input": "Hello, how are you?",
            "output": "Bonjour, comment allez-vous?"
        }
    ]
    """
    (tmp_path / "alpaca.json").write_text(alpaca_data)

    config = lt.PipelineConfig.from_dataset(
        model_repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        file_path=tmp_path / "alpaca.json",
    )

    # Run finetuning to get a model adapter
    config.trainer.max_epochs = 1
    config.trainer.evaluation.do_eval = True
    adapter_path = lt.run_finetuning(config)

    with patch("lightning_tune.evaluation.evaluate.load") as mock_load:
        lt.run_evaluation(config, str(adapter_path))
        mock_load.assert_called()
