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
    {"db_instruction": "What is ML?", "db_output": "ML is Machine Learning."}
]).to_sql("train_data", conn, index=False, if_exists="replace")
conn.close()

csv_path = "test.csv"
pd.DataFrame([
    {"csv_instruction": "Translate Hola.", "csv_output": "Hello."},
    {"csv_instruction": "Translate Adios.", "csv_output": "Goodbye."}
]).to_csv(csv_path, index=False)


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
            ),
            DatasetConfig(
                file_path=Path(csv_path),
                instruction_column="csv_instruction",
                output_column="csv_output",
                input_column=""
            )
        ]
    )
)

def format_prompt_db(x):
    return {"text": f"### Instruction:\n{x['db_instruction']}\n\n### Response:\n{x['db_output']}"}

def format_prompt_csv(x):
    return {"text": f"### Instruction:\n{x['csv_instruction']}\n\n### Response:\n{x['csv_output']}"}

df_db = pl.read_database_uri(query="SELECT * FROM train_data", uri=f"sqlite://{db_path}", engine="connectorx")
ds1 = HFDataset(pa.Table.from_batches(df_db.to_arrow().to_batches()))

from datasets import load_dataset
ds2 = load_dataset("csv", data_files=csv_path, split="train")

f = Features({"text": Value("string")})

ds1_list = [format_prompt_db(x) for x in ds1]
ds1 = HFDataset.from_list(ds1_list, features=f)

ds2_list = [format_prompt_csv(x) for x in ds2]
ds2 = HFDataset.from_list(ds2_list, features=f)

ds = concatenate_datasets([ds1, ds2])
print("NEW MERGED LEN", len(ds))

