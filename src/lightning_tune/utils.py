import torch
import sqlite3
import json
from datetime import datetime

def get_auto_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def init_db():
    conn = sqlite3.connect("jobs.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS job_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT,
            step INTEGER,
            epoch REAL,
            loss REAL,
            reward REAL,
            extra_metrics TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def log_job_metrics(job_id, step, epoch, loss, reward=None, extra_metrics=None):
    init_db()
    conn = sqlite3.connect("jobs.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO job_metrics (job_id, step, epoch, loss, reward, extra_metrics)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (job_id, step, epoch, loss, reward, json.dumps(extra_metrics or {})))
    conn.commit()
    conn.close()
