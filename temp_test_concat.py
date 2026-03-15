import sqlite3
import pandas as pd
from pathlib import Path
from datasets import Dataset as HFDataset, concatenate_datasets
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

dataset_dict = prepare_text_dataset(config)
train_ds = dataset_dict["train"]
print(f"LEN: {len(train_ds)}")
print(train_ds["text"])
import sqlite3
import pandas as pd
from pathlib import Path
from datasets import Dataset as HFDataset, concatenate_datasets
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

dataset_dict = prepare_text_dataset(config)
train_ds = dataset_dict["train"]
print(f"LEN: {len(train_ds)}")
for row in train_ds:
    print(row)
import sqlite3
import pandas as pd
from pathlib import Path
from datasets import Dataset as HFDataset, concatenate_datasets
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

from datasets import Features, Value

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

# Simulate full manual processing
def format_prompt_for_dataset(ds_config, global_config):
    input_col = getattr(ds_config, "input_column", None) or global_config.input_column
    instr_col = getattr(ds_config, "instruction_column", None) or global_config.instruction_column
    out_col = getattr(ds_config, "output_column", None) or global_config.output_column

    def _format(example):
        if input_col and example.get(input_col):
            return {
                "text": f"### Instruction:\n{example.get(instr_col, '')}\n\n### Input:\n{example.get(input_col, '')}\n\n### Response:\n{example.get(out_col, '')}"
            }
        return {
            "text": f"### Instruction:\n{example.get(instr_col, '')}\n\n### Response:\n{example.get(out_col, '')}"
        }
    return _format

all_items = []
for src in config.data.datasets:
    formatter = format_prompt_for_dataset(src, config.data)
    if src.db_uri:
        df = pl.read_database_uri(query=src.db_query, uri=src.db_uri, engine="connectorx")
        ds = HFDataset(pa.Table.from_batches(df.to_arrow().to_batches()))
        all_items.extend([formatter(x) for x in ds])
    elif src.file_path:
        from datasets import load_dataset
        ds = load_dataset('csv', data_files=str(src.file_path), split="train")
        all_items.extend([formatter(x) for x in ds])

print(f"RAW ITEMS ACCUMULATED: {len(all_items)}")
for i in all_items: print(i)

final_ds = HFDataset.from_list(all_items)
print(f"FINAL DS LEN: {len(final_ds)}")
import sqlite3
import pandas as pd
from pathlib import Path
from datasets import Dataset as HFDataset, concatenate_datasets
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

dataset_dict = prepare_text_dataset(config)
train_ds = dataset_dict["train"]
print(f"LEN: {len(train_ds)}")
for row in train_ds:
    print(row)
import sqlite3
import pandas as pd
from pathlib import Path
from datasets import Dataset as HFDataset, concatenate_datasets
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


all_items = []
sources = config.data.datasets
    
for src in sources:
    input_col = getattr(src, "input_column", None)
    instr_col = getattr(src, "instruction_column", None)
    out_col = getattr(src, "output_column", None)
    
    def _format(example):
        # Force string casting immediately to drop any pyarrow/pandas latent metadata
        i_txt = str(example.get(instr_col, '')) if instr_col in example else ''
        in_txt = str(example.get(input_col, '')) if input_col and input_col in example else ''
        o_txt = str(example.get(out_col, '')) if out_col in example else ''

        if in_txt:
            return {"text": f"### Instruction:\n{i_txt}\n\n### Input:\n{in_txt}\n\n### Response:\n{o_txt}"}
        return {"text": f"### Instruction:\n{i_txt}\n\n### Response:\n{o_txt}"}

    if src.db_uri:
         df = pl.read_database_uri(query=src.db_query, uri=src.db_uri, engine="connectorx")
         ds = HFDataset.from_pandas(df.to_pandas())
         all_items.extend([_format(x) for x in ds])
    elif src.file_path:
         from datasets import load_dataset
         ds = load_dataset('csv', data_files=str(src.file_path), split="train")
         all_items.extend([_format(x) for x in ds])

print(f"RAW ITEMS: {len(all_items)}")
for i in all_items: print(i)

dataset = HFDataset.from_list(all_items)
print(f"FINISHED DS: {len(dataset)}")
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

all_items = []
def format_prompt_for_dataset(ds_config, global_config):
    input_col = getattr(ds_config, "input_column", None) or getattr(global_config, "input_column", None)
    instr_col = getattr(ds_config, "instruction_column", None) or getattr(global_config, "instruction_column", None)
    out_col = getattr(ds_config, "output_column", None) or getattr(global_config, "output_column", None)

    def _format(example):
        i_txt = str(example.get(instr_col, '')) if instr_col in example else ''
        in_txt = str(example.get(input_col, '')) if input_col and input_col in example else ''
        o_txt = str(example.get(out_col, '')) if out_col in example else ''

        if in_txt:
            return dict(text=f"### Instruction:\n{i_txt}\n\n### Input:\n{in_txt}\n\n### Response:\n{o_txt}")
        return dict(text=f"### Instruction:\n{i_txt}\n\n### Response:\n{o_txt}")
    return _format

for src in config.data.datasets:
    formatter = format_prompt_for_dataset(src, config.data)
    if src.db_uri:
         df = pl.read_database_uri(query=src.db_query, uri=src.db_uri, engine="connectorx")
         records = df.to_pandas().to_dict('records')
         all_items.extend([formatter(x) for x in records])
         print(f"Added {len(records)} DB records")
    elif src.file_path:
         df = pd.read_csv(src.file_path)
         records = df.to_dict('records')
         all_items.extend([formatter(x) for x in records])
         print(f"Added {len(records)} CSV records")


print(f"RAW ITEMS: {len(all_items)}")
for r in all_items: print(r)
dataset = HFDataset.from_list(all_items)
print(f"FINAL LENGTH: {len(dataset)}")

