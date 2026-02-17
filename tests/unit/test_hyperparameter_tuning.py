import lightning_tune as lt
from pathlib import Path

def test_hyperparameter_tuning(tmp_path):
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

    lt.run_hyperparameter_tuning(config, n_trials=1)
