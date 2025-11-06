import optuna
from .config import PipelineConfig
from .train import run_finetuning

def _objective(trial, config: PipelineConfig):
    # Define the search space for the hyperparameters
    config.train.llm_lr = trial.suggest_float("llm_lr", 1e-6, 1e-4, log=True)
    config.train.peft.r = trial.suggest_int("r", 4, 32, step=4)
    config.train.peft.lora_alpha = trial.suggest_int("lora_alpha", 8, 64, step=8)

    # Run the finetuning process
    trainer = run_finetuning(config)

    # Return the validation loss
    if hasattr(trainer, "state") and hasattr(trainer.state, "best_metric"):
        return trainer.state.best_metric
    else:
        # If the trainer does not have a state attribute, it means that the
        # training was not successful. In this case, we return a large value
        # to indicate that this trial should be pruned.
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
