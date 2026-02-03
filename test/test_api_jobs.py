from fastapi.testclient import TestClient
from app.api.main import app
from unittest.mock import patch

client = TestClient(app)

def test_start_train():
    with patch("app.api.main.job_manager.start_training_job") as mock_start:
        mock_start.return_value = "job-123"

        response = client.post(
            "/train",
            json={"config": {"model": {"repo_id": "foo"}, "data": {"dataset_repo_id": "bar", "text_columns": ["text"], "output_column": "out", "split": "train"}}}
        )
        assert response.status_code == 200
        assert response.json()["job_id"] == "job-123"

def test_stream_logs():
    with patch("app.api.main.job_manager.stream_logs") as mock_stream:
        # Mock generator
        async def mock_gen(job_id):
            yield "Log line 1"
            yield "Log line 2"

        mock_stream.side_effect = mock_gen

        with client.websocket_connect("/train/job-123/logs") as websocket:
            data = websocket.receive_text()
            assert data == "Log line 1"
            data = websocket.receive_text()
            assert data == "Log line 2"
