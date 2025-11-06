import logging
from .config import (
    PipelineConfig,
    ModelConfig,
    DataConfig,
    TrainerConfig,
    TrainConfig,
    PeftConfig,
    TabularConfig,
    VisionConfig,
    EvaluationConfig,
    DeploymentConfig,
)
from .train import run_finetuning
from .deploy import launch_server
from .hyperparameter_tuning import run_hyperparameter_tuning
from .evaluation import run_evaluation

# Set up a logger for the library
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
