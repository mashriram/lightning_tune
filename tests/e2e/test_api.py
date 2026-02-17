import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.api.main import app

client = TestClient(app)

def test_analyze_multiset_support():
    # Mock PipelineConfig.from_dataset to avoid actual HF calls or heavy processing
    with patch("src.lightning_tune.config.PipelineConfig.from_dataset") as mock_analyze:
        mock_analyze.return_value = {"status": "ok", "config": {}}
        
        payload = {
            "model_repo_id": "google/gemma-2b",
            "datasets": [
                {"repo_id": "glue", "config_name": "mrpc", "split": "train"},
                {"file_path": "/tmp/dummy.csv", "split": "train"}
            ]
        }
        
        response = client.post("/analyze", json=payload)
        assert response.status_code == 200
        
        # Verify arguments passed to from_dataset
        mock_analyze.assert_called_once()
        _, kwargs = mock_analyze.call_args
        assert "datasets" in kwargs
        assert len(kwargs["datasets"]) == 2
        assert kwargs["datasets"][0]["repo_id"] == "glue"

def test_analyze_legacy_support():
    with patch("src.lightning_tune.config.PipelineConfig.from_dataset") as mock_analyze:
        mock_analyze.return_value = {"status": "ok"}
        
        payload = {
            "model_repo_id": "google/gemma-2b",
            "dataset_repo_id": "glue",
            "split": "train"
        }
        
        response = client.post("/analyze", json=payload)
        assert response.status_code == 200
        
        mock_analyze.assert_called_once()
        _, kwargs = mock_analyze.call_args
        assert kwargs.get("dataset_repo_id") == "glue"
        assert "datasets" not in kwargs

def test_analyze_missing_input():
    response = client.post("/analyze", json={"model_repo_id": "foo"})
    assert response.status_code == 400
    assert "Provide at least one dataset source" in response.json()["detail"]

def test_job_management():
    # Mock job_manager
    with patch("app.api.main.job_manager") as mock_jm:
        mock_jm.start_training_job.return_value = "job-123"
        mock_jm.get_job_status.return_value = "running"
        mock_jm.stop_job.return_value = True

        # Start Train
        resp = client.post("/train", json={"config": {"model": {"repo_id": "foo"}}})
        assert resp.status_code == 200
        assert resp.json()["job_id"] == "job-123"

        # Get Status
        resp = client.get("/jobs/job-123")
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"

        # Stop Job
        resp = client.delete("/jobs/job-123")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

def test_serve_endpoint():
    with patch("app.api.main.job_manager") as mock_jm:
        mock_jm.start_serving_job.return_value = "456"
        
        payload = {
            "job_id": "job-123",
            "port": 8001,
            "use_vllm": True
        }
        resp = client.post("/serve", json=payload)
        assert resp.status_code == 200
        assert "service_456" in resp.json()["job_id"]
        assert resp.json()["service_url"] == "http://localhost:8001"
