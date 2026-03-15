import sqlite3
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory
from datasets import Dataset as HFDataset, concatenate_datasets
import pyarrow as pa
import polars as pl

db_path = "test.db"
conn = sqlite3.connect(db_path)
pd.DataFrame([
    {"db_instruction": "What is AI?", "db_output": "AI is Artificial Intelligence."},
    {"db_instruction": "What is ML?", "db_output": "ML is Machine Learning."}
]).to_sql("train_data", conn, index=False, if_exists="replace")
conn.close()

df = pl.read_database_uri(query="SELECT * FROM train_data", uri=f"sqlite://{db_path}", engine="connectorx")
print("POLARS DF LEN:", len(df))
ds1 = HFDataset(pa.Table.from_batches(df.to_arrow().to_batches()))
print("HF DATASET 1 LEN:", len(ds1))

ds1 = HFDataset.from_list([{"text": x["db_instruction"] + x["db_output"]} for x in ds1])
print("POST MAP HF DATASET 1:", len(ds1))


csv_path = "test.csv"
pd.DataFrame([
    {"csv_instruction": "Translate Hola.", "csv_output": "Hello."},
    {"csv_instruction": "Translate Adios.", "csv_output": "Goodbye."}
]).to_csv(csv_path, index=False)
from datasets import load_dataset
ds2 = load_dataset("csv", data_files=csv_path, split="train")
print("HF DATASET 2 LEN:", len(ds2))

ds2 = HFDataset.from_list([{"text": x["csv_instruction"] + x["csv_output"]} for x in ds2])
print("POST MAP HF DATASET 2:", len(ds2))

merged = concatenate_datasets([ds1, ds2])
print("MERGED LEN:", len(merged))
