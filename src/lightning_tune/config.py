import logging, warnings
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict
from pathlib import Path
import polars as pl
from transformers import AutoConfig


class ModelConfig(BaseModel):
    repo_id: str
    base_model_dir: Path = Path("models/")


class PeftConfig(BaseModel):
    method: Literal["qlora", "lora", "dora"] = "qlora"
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05


class TabularConfig(BaseModel):
    categorical_columns: List[str]
    numerical_columns: List[str]
    projection_dim: int
    categorical_cardinality: Optional[Dict[str, int]] = None


class VisionConfig(BaseModel):
    image_column: str
    model_name: str = "vit_base_patch16_224"
    projection_dim: int


class DataConfig(BaseModel):
    file_path: Path
    instruction_column: str = "instruction"
    input_column: str = "input"
    output_column: str = "output"
    text_columns: List[str] = Field(default_factory=list)
    image_root_path: Optional[Path] = None
    max_seq_length: int = 512
    tabular_config: Optional[TabularConfig] = None
    vision_config: Optional[VisionConfig] = None


class EvaluationConfig(BaseModel):
    do_eval: bool = True
    eval_dataset_size: float = 0.1


from .utils import get_auto_device

class TrainerConfig(BaseModel):
    max_epochs: int = 3
    devices: int = 1
    device: str = Field(default_factory=get_auto_device)
    logger: Literal["csv", "tensorboard"] = "tensorboard"
    checkpoint_callback: bool = True
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


class TrainConfig(BaseModel):
    peft: PeftConfig = Field(default_factory=PeftConfig)
    llm_lr: float = 5e-5
    tower_lr: float = 1e-4
    batch_size: int = 2


class DeploymentConfig(BaseModel):
    port: int = 8000


class PipelineConfig(BaseModel):
    model: ModelConfig
    data: DataConfig
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)

    @property
    def is_multimodal(self) -> bool:
        return (
            self.data.vision_config is not None or self.data.tabular_config is not None
        )

    def get_output_dir(self) -> Path:
        base = self.model.repo_id.replace("/", "_")
        mode = "multimodal" if self.is_multimodal else self.train.peft.method
        return Path("finetuning_runs") / f"{base}_{mode}"

    @staticmethod
    def _analyze_schema(df_sample: pl.DataFrame) -> Dict:
        analysis = {
            "image_column": None,
            "text_columns": [],
            "numerical_columns": [],
            "categorical_columns": [],
            "output_column": None,
        }
        potential_output_cols = ("output", "target", "label", "completion", "review")
        unstructured_cols = []

        for col, dtype in df_sample.schema.items():
            l_col = col.lower()
            if ("image" in l_col or "path" in l_col) and dtype == pl.Utf8:
                analysis["image_column"] = col
            elif dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                analysis["numerical_columns"].append(col)
            elif dtype == pl.Utf8:
                unstructured_cols.append(col)

        # Prioritize finding the output column among all unstructured columns
        for col_name in potential_output_cols:
            if col_name in unstructured_cols:
                analysis["output_column"] = col_name
                unstructured_cols.remove(col_name)
                break

        # Classify remaining unstructured columns
        common_text_cols = ("text", "content", "description", "comment", "message", "body")
        for col in unstructured_cols:
            l_col = col.lower()
            avg_str_len = df_sample[col].str.len_chars().mean()
            # If low cardinality and short strings, it's likely categorical
            if any(c in l_col for c in common_text_cols):
                analysis["text_columns"].append(col)
            elif df_sample[col].n_unique() < 50 and avg_str_len < 30:
                analysis["categorical_columns"].append(col)
            else:
                analysis["text_columns"].append(col)

        # Fallback for output column if not explicitly found
        if not analysis["output_column"]:
            if analysis["text_columns"]:
                # Prefer the text column with the longest average string length as output
                longest_text_col = max(
                    analysis["text_columns"],
                    key=lambda c: df_sample[c].str.len_chars().mean(),
                )
                analysis["output_column"] = longest_text_col
                analysis["text_columns"].remove(longest_text_col)
                warnings.warn(
                    f"Could not automatically determine the output column. Using '{analysis['output_column']}' as the target."
                )
            elif analysis["categorical_columns"]:
                # As a last resort, use a categorical column
                analysis["output_column"] = analysis["categorical_columns"].pop()
                warnings.warn(
                    f"Could not find a clear text output column. Using categorical column '{analysis['output_column']}' as the target."
                )

        return analysis

    @staticmethod
    def _suggest_hyperparameters(dataset_size: int) -> Dict:
        if dataset_size < 10000:
            return {"llm_lr": 5e-5, "epochs": 3, "r": 8}
        elif dataset_size < 100000:
            return {"llm_lr": 3e-5, "epochs": 2, "r": 16}
        else:
            return {"llm_lr": 2e-5, "epochs": 1, "r": 32}

    @staticmethod
    def _read_and_sample_dataset(file_path: Path) -> pl.DataFrame:
        file_suffix = file_path.suffix.lower()
        if file_suffix == ".csv":
            df = pl.read_csv(file_path)
        elif file_suffix in (".json", ".jsonl"):
            df = pl.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_suffix}")
        return df, df.head(100)

    @classmethod
    def from_dataset(
        cls, model_repo_id: str, file_path: Path, **kwargs
    ) -> "PipelineConfig":
        logging.info(f"🔬 Analyzing dataset '{file_path}' to create smart configuration...")
        try:
            df, df_sample = cls._read_and_sample_dataset(file_path)
        except Exception as e:
            logging.error(f"Failed to read or process the dataset: {e}")
            raise

        is_alpaca_format = all(c in df.columns for c in ["instruction", "input", "output"])

        if is_alpaca_format:
            schema_analysis = {
                "image_column": None,
                "text_columns": [],
                "numerical_columns": [],
                "categorical_columns": [],
                "output_column": "output",
                "instruction_column": "instruction",
                "input_column": "input",
            }
        else:
            schema_analysis = cls._analyze_schema(df_sample)
        hyperparams = cls._suggest_hyperparameters(df.height)

        llm_hf_config = AutoConfig.from_pretrained(model_repo_id)
        data_cfg = DataConfig(
            file_path=file_path,
            text_columns=schema_analysis["text_columns"],
            output_column=schema_analysis["output_column"],
            image_root_path=kwargs.pop("image_root_path", None),
            **{
                k: v
                for k, v in schema_analysis.items()
                if k in ("instruction_column", "input_column")
            },
        )

        if schema_analysis.get("image_column"):
            data_cfg.vision_config = VisionConfig(
                image_column=schema_analysis["image_column"],
                projection_dim=llm_hf_config.hidden_size,
            )
        if (
            schema_analysis["numerical_columns"]
            or schema_analysis["categorical_columns"]
        ):
            data_cfg.tabular_config = TabularConfig(
                numerical_columns=schema_analysis["numerical_columns"],
                categorical_columns=schema_analysis["categorical_columns"],
                projection_dim=llm_hf_config.hidden_size,
            )

        train_cfg = TrainConfig(
            llm_lr=hyperparams["llm_lr"], peft=PeftConfig(r=hyperparams["r"])
        )
        trainer_cfg = TrainerConfig(max_epochs=hyperparams["epochs"])

        logging.info(
            f"✅ Analysis complete. Smart Hyperparameters: "
            f"LR={hyperparams['llm_lr']}, Epochs={hyperparams['epochs']}, LoRA Rank={hyperparams['r']}"
        )
        return cls(
            model=ModelConfig(repo_id=model_repo_id),
            data=data_cfg,
            train=train_cfg,
            trainer=trainer_cfg,
            **kwargs,
        )
