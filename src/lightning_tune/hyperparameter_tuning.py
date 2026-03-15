import optuna
import logging
from .config import PipelineConfig
from .train import run_finetuning

def _objective(trial, config: PipelineConfig):
    # Define the search space for the hyperparameters
    config.train.llm_lr = trial.suggest_float("llm_lr", 1e-6, 1e-4, log=True)
    config.train.peft.r = trial.suggest_int("r", 4, 32, step=4)
    config.train.peft.lora_alpha = trial.suggest_int("lora_alpha", 8, 64, step=8)

    # Run the finetuning process
    result = run_finetuning(config)
    metrics = result.get("metrics", {})

    # Return the validation loss (SFTTrainer uses 'eval_loss', MultiModal uses 'val_loss')
    val_loss = metrics.get("eval_loss") or metrics.get("val_loss")
    
    if val_loss is not None:
        return val_loss
    else:
        # If no loss found, return inf to indicate failure/unoptimal
        logging.warning("No evaluation loss found in metrics. Trial marked as infinity.")
        return float("inf")

def run_hyperparameter_tuning(config: PipelineConfig, n_trials: int = 10):
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(lambda trial: _objective(trial, config), n_trials=n_trials)

    logging.info(f"HPT Finished. Number of trials: {len(study.trials)}")
    if not study.trials:
        logging.warning("No trials completed. Config remains unchanged.")
        return config

    trial = study.best_trial
    logging.info(f"Best Trial Value: {trial.value}")
    
    # Update the config with the best hyperparameters
    if "llm_lr" in trial.params:
        config.train.llm_lr = trial.params["llm_lr"]
    if "r" in trial.params:
        config.train.peft.r = trial.params["r"]
    if "lora_alpha" in trial.params:
        config.train.peft.lora_alpha = trial.params["lora_alpha"]
    
    logging.info(f"Updated config with best params: {trial.params}")
    return config
