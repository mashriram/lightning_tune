import typer
from pathlib import Path
from .config import PipelineConfig
from .hyperparameter_tuning import run_hyperparameter_tuning
from .train import run_finetuning
from .evaluation import run_evaluation
from .deploy import launch_server

app = typer.Typer()


@app.command()
def tune(config_path: str, n_trials: int = 10):
    config = PipelineConfig.from_yaml(config_path)
    run_hyperparameter_tuning(config, n_trials=n_trials)

@app.command()
def evaluate(config_path: str, model_path: str):
    config = PipelineConfig.from_yaml(config_path)
    run_evaluation(config, model_path)

@app.command()
def train(config_path: str):
    config = PipelineConfig.from_yaml(config_path)
    run_finetuning(config)

@app.command()
def serve(config_path: str, model_path: str):
    config = PipelineConfig.from_yaml(config_path)
    launch_server(config, Path(model_path))

if __name__ == "__main__":
    app()
