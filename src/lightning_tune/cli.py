import typer
from .config import PipelineConfig
from .hyperparameter_tuning import run_hyperparameter_tuning

app = typer.Typer()

from .evaluation import run_evaluation

@app.command()
def tune(config_path: str, n_trials: int = 10):
    config = PipelineConfig.from_yaml(config_path)
    run_hyperparameter_tuning(config, n_trials=n_trials)

@app.command()
def evaluate(config_path: str, model_path: str):
    config = PipelineConfig.from_yaml(config_path)
    run_evaluation(config, model_path)

if __name__ == "__main__":
    app()
