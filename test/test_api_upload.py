from fastapi.testclient import TestClient
from app.api.main import app
from unittest.mock import patch, ANY
import io

client = TestClient(app)

def test_upload_file():
    # Create mock file
    file_content = b"header1,header2\nval1,val2"
    files = {"file": ("test.csv", file_content, "text/csv")}

    response = client.post("/upload", files=files)
    assert response.status_code == 200
    assert "file_path" in response.json()
    assert response.json()["file_path"].endswith(".csv")

def test_analyze_with_file_path():
    # Mock config
    with patch("app.api.main.PipelineConfig.from_dataset") as mock_from_ds:
        mock_from_ds.return_value.model_dump.return_value = {"model": "test"}

        response = client.post(
            "/analyze",
            json={"model_repo_id": "foo", "file_path": "/tmp/test.csv"}
        )
        assert response.status_code == 200
        # Check that file_path was passed to PipelineConfig
        call_kwargs = mock_from_ds.call_args[1]
        assert str(call_kwargs["file_path"]) == "/tmp/test.csv"
