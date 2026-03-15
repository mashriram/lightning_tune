import sqlite3
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory
from datasets import Dataset as HFDataset, concatenate_datasets, Features, Value
import pyarrow as pa
import polars as pl
from lightning_tune.data import prepare_text_dataset
from lightning_tune.config import PipelineConfig, DatasetConfig, DataConfig, ModelConfig

db_path = "test.db"
conn = sqlite3.connect(db_path)
pd.DataFrame([
    {"db_instruction": "What is AI?", "db_output": "AI is Artificial Intelligence."},
    {"db_instruction": "What is ML?", "db_output": "ML is Machine Learning."} # The missing row!
]).to_sql("train_data", conn, index=False, if_exists="replace")
conn.close()

config = PipelineConfig(
    model=ModelConfig(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    data=DataConfig(
        datasets=[
            DatasetConfig(
                db_uri=f"sqlite://{db_path}",
                db_query="SELECT * FROM train_data",
                instruction_column="db_instruction",
                output_column="db_output",
                input_column=""
            )
        ]
    )
)

dataset_dict = prepare_text_dataset(config)
train_ds = dataset_dict["train"]
print(f"LEN: {len(train_ds)}")
for row in train_ds:
    print(row)
