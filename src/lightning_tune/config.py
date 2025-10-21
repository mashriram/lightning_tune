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


class TrainerConfig(BaseModel):
    max_epochs: int = 3
    precision: str = "bf16"
    devices: int = 1
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

    @classmethod
    def from_dataset(
        cls, model_repo_id: str, file_path: Path, **kwargs
    ) -> "PipelineConfig":
        print(f"🔬 Analyzing dataset '{file_path}' to create smart configuration...")
        if file_path.suffix.lower() != ".csv":
            raise ValueError(f"Invalid file type. Please upload a CSV file.")

        df = pl.read_csv(file_path)
        dataset_size = df.height
        df_sample = df.head(100)
        img_col, txt_cols, num_cols, cat_cols = None, [], [], []
        for col, dtype in df_sample.schema.items():
            l_col = col.lower()
            if ("image" in l_col or "path" in l_col) and dtype == pl.Utf8:
                img_col = col
            elif dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                num_cols.append(col)
            elif dtype == pl.Utf8 and df_sample[col].n_unique() < 50:
                cat_cols.append(col)
            elif dtype == pl.Utf8:
                txt_cols.append(col)
        out_col = txt_cols.pop() if txt_cols else "output"

        llm_hf_config = AutoConfig.from_pretrained(model_repo_id)
        data_cfg = DataConfig(
            file_path=file_path,
            text_columns=txt_cols,
            output_column=out_col,
            image_root_path=kwargs.pop("image_root_path", None),
        )

        if img_col:
            data_cfg.vision_config = VisionConfig(
                image_column=img_col, projection_dim=llm_hf_config.hidden_size
            )
        if num_cols or cat_cols:
            data_cfg.tabular_config = TabularConfig(
                numerical_columns=num_cols,
                categorical_columns=cat_cols,
                projection_dim=llm_hf_config.hidden_size,
            )

        llm_lr, epochs, r = (
            (5e-5, 3, 8)
            if dataset_size < 10000
            else ((3e-5, 2, 16) if dataset_size < 100000 else (2e-5, 1, 32))
        )
        train_cfg = TrainConfig(llm_lr=llm_lr, peft=PeftConfig(r=r))
        trainer_cfg = TrainerConfig(max_epochs=epochs)

        print(
            f"✅ Analysis complete.\n\t- Smart Hyperparameters: LR={llm_lr}, Epochs={epochs}, LoRA Rank={r}"
        )
        return cls(
            model=ModelConfig(repo_id=model_repo_id),
            data=data_cfg,
            train=train_cfg,
            trainer=trainer_cfg,
            **kwargs,
        )
