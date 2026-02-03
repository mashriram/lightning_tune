from fastapi.testclient import TestClient
from app.api.main import app
from unittest.mock import patch

client = TestClient(app)

def test_search_models():
    # Mock search_models to avoid network call
    with patch("app.api.main.search_models") as mock_search:
        mock_search.return_value = [{"id": "test/model", "downloads": 100, "likes": 10, "private": False, "task": "text-generation"}]
        response = client.get("/models?query=test")
        assert response.status_code == 200
        assert response.json()[0]["id"] == "test/model"

def test_analyze_no_auth_needed():
    # Mock PipelineConfig.from_dataset
    with patch("app.api.main.PipelineConfig.from_dataset") as mock_from_dataset:
        mock_from_dataset.return_value.model_dump.return_value = {"model": {"repo_id": "foo"}}

        response = client.post(
            "/analyze",
            json={"model_repo_id": "foo", "dataset_repo_id": "bar"}
        )
        assert response.status_code == 200
        assert response.json()["model"]["repo_id"] == "foo"
