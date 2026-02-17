from fastapi.testclient import TestClient
from app.api.main import app
from unittest.mock import patch

client = TestClient(app)

def test_serve_model():
    with patch("app.api.main.job_manager.start_serving_job") as mock_serve:
        mock_serve.return_value = "123"

        response = client.post(
            "/serve",
            json={"model_path": "/tmp/model", "config": {"deployment": {"port": 8000}}}
        )
        assert response.status_code == 200
        assert response.json()["job_id"] == "service_123"
        assert response.json()["service_url"] == "http://localhost:8000"
