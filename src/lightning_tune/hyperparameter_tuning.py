import optuna
from .config import PipelineConfig
from .train import run_finetuning

import json
def _objective(trial, config: PipelineConfig):
    # Define the search space for the hyperparameters
    config.train.llm_lr = trial.suggest_float("llm_lr", 1e-6, 1e-4, log=True)
    config.train.peft.r = trial.suggest_int("r", 4, 32, step=4)
    config.train.peft.lora_alpha = trial.suggest_int("lora_alpha", 8, 64, step=8)

    # Run the finetuning process
    run_finetuning(config, trial)

    # Return the validation loss
    metrics_path = config.get_output_dir() / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        return metrics.get("eval_loss", float("inf"))
    return float("inf")

def run_hyperparameter_tuning(config: PipelineConfig, n_trials: int = 10):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: _objective(trial, config), n_trials=n_trials)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # TODO: Save the best hyperparameters to a file
