import sqlite3
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory
from lightning_tune.config import PipelineConfig, DatasetConfig, DataConfig, ModelConfig

def test_sqlite_interleave_pipeline():
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        db_path = tmp_path / "test.db"
        csv_path = tmp_path / "test.csv"
        
        # 1. Setup SQLite
        conn = sqlite3.connect(db_path)
        pd.DataFrame([
            {"db_instruction": "What is AI?", "db_output": "AI is Artificial Intelligence."},
            {"db_instruction": "What is ML?", "db_output": "ML is Machine Learning."}
        ]).to_sql("train_data", conn, index=False)
        conn.close()
        
        # 2. Setup CSV
        pd.DataFrame([
            {"csv_instruction": "Translate Hola.", "csv_output": "Hello."},
            {"csv_instruction": "Translate Adios.", "csv_output": "Goodbye."}
        ]).to_csv(csv_path, index=False)
        
        # 3. Configure Interleaved Pipeline
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
                        file_path=csv_path,
                        instruction_column="csv_instruction",
                        output_column="csv_output",
                        input_column=""
                    )
                ]
            ),
            trainer=dict(evaluation=dict(do_eval=False))
        )
        
        # 4. Process Dataset
        from lightning_tune.data import prepare_text_dataset
        dataset_dict = prepare_text_dataset(config)
        
        train_ds = dataset_dict["train"]
        
        # Verify length and interleaving (concatenation)
        assert len(train_ds) == 4
        
        # Verify formatting closure mapped variables correctly
        texts = [row["text"] for row in train_ds]
        assert "### Instruction:\nWhat is AI?\n\n### Response:\nAI is Artificial Intelligence." in texts
        assert "### Instruction:\nTranslate Hola.\n\n### Response:\nHello." in texts
