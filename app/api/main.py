import sys
from pathlib import Path

# Add project root to sys.path to allow absolute imports
# matching the package structure (app.api...)
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, Header, Depends, WebSocket, UploadFile, File
from typing import List, Optional, Union, Dict, Any
import shutil
import uuid
import logging

# Use absolute imports
from app.api.schemas import SearchResult, DatasetSearchResult, AnalyzeRequest, TrainRequest, JobResponse, ServeRequest, PushRequest
from app.api.job_manager import job_manager
from src.lightning_tune.hf_utils import search_models, search_datasets
from src.lightning_tune.config import PipelineConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(title="Lightning Tune API")

UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def get_token(authorization: Optional[str] = Header(None)) -> Optional[str]:
    if authorization:
        if authorization.startswith("Bearer "):
            return authorization.split(" ")[1]
        return authorization
    return None

@app.get("/models", response_model=List[SearchResult])
def find_models(query: str, limit: int = 20, token: Optional[str] = Depends(get_token)):
    try:
        results = search_models(query, limit, token)
        return results
    except Exception as e:
        logger.error(f"Search models failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/datasets", response_model=List[DatasetSearchResult])
def find_datasets(query: str, limit: int = 20, token: Optional[str] = Depends(get_token)):
    try:
        results = search_datasets(query, limit, token)
        return results
    except Exception as e:
        logger.error(f"Search datasets failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a dataset file (CSV/JSON).
    """
    try:
        file_ext = Path(file.filename).suffix
        file_id = str(uuid.uuid4())
        file_path = UPLOAD_DIR / f"{file_id}{file_ext}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {"file_path": str(file_path.absolute())}
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
def analyze_dataset(request: AnalyzeRequest, token: Optional[str] = Depends(get_token)):
    """
    Analyze a HF dataset or uploaded file and suggest configuration.
    """
    try:
        # Check if file path or repo id
        # Check inputs
        kwargs = {}
        if request.datasets:
            kwargs["datasets"] = request.datasets

        if request.file_path:
            kwargs["file_path"] = Path(request.file_path)
            # Legacy single file handling logic if needed
            if request.file_path.endswith(".zip"):
                 pass
        elif request.dataset_repo_id:
            kwargs["dataset_repo_id"] = request.dataset_repo_id
            kwargs["split"] = request.split
        
        if not kwargs:
            raise HTTPException(status_code=400, detail="Provide at least one dataset source (datasets list, dataset_repo_id, or file_path)")

        result = PipelineConfig.from_dataset(
            model_repo_id=request.model_repo_id,
            token=token,
            **kwargs
        )
        if isinstance(result, dict):
             return result
        return result.model_dump()
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        msg = str(e)
        if "401" in msg or "403" in msg:
             raise HTTPException(status_code=401, detail="Hugging Face authentication required. Please provide a valid token in 'Authorization' header.")
        raise HTTPException(status_code=400, detail=f"Analysis failed: {str(e)}")

@app.post("/train", response_model=JobResponse)
def start_train(request: TrainRequest, token: Optional[str] = Depends(get_token)):
    """
    Start a training job.
    """
    try:
        job_id = job_manager.start_training_job(request.config, token)
        return JobResponse(job_id=job_id, status="running", output_dir=f"jobs/{job_id}")
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job_status_endpoint(job_id: str, token: Optional[str] = Depends(get_token)):
    """
    Get the status of a job.
    """
    status = job_manager.get_job_status(job_id)
    return JobResponse(job_id=job_id, status=status)

@app.delete("/jobs/{job_id}")
def stop_job_endpoint(job_id: str, token: Optional[str] = Depends(get_token)):
    """
    Stop a running job.
    """
    success = job_manager.stop_job(job_id)
    if success:
        return {"status": "stopped", "job_id": job_id}
    else:
        raise HTTPException(status_code=404, detail="Job not found or not running.")

@app.post("/tune", response_model=JobResponse)
def start_tune(request: TrainRequest, n_trials: int = 10, token: Optional[str] = Depends(get_token)):
    """
    Start a hyperparameter tuning job.
    """
    try:
        job_id = job_manager.start_tuning_job(request.config, n_trials, token)
        return JobResponse(job_id=job_id, status="running", output_dir=f"jobs/{job_id}")
    except Exception as e:
        logger.error(f"Failed to start tuning: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/adapters")
def list_adapters():
    """
    List all available fine-tuned adapters (finished jobs).
    """
    adapters = []
    if not Path("jobs").exists():
        return []
    for job_dir in Path("jobs").iterdir():
        if job_dir.is_dir():
            if (job_dir / "final_adapter").exists():
                adapters.append({"name": job_dir.name, "path": str((job_dir / "final_adapter").absolute())})
            elif (job_dir / "best_model.ckpt").exists():
                adapters.append({"name": job_dir.name, "path": str(job_dir.absolute())})
    return adapters

@app.post("/serve", response_model=JobResponse)
def serve_model(request: ServeRequest, token: Optional[str] = Depends(get_token)):
    """
    Serve a trained model.
    """
    try:
        service_id = job_manager.start_serving_job(
            job_id=request.job_id,
            model_path=request.model_path,
            config=request.config,
            port=request.port,
            use_vllm=request.use_vllm,
            hf_token=token
        )
        full_id = f"service_{service_id}"
        return JobResponse(
            job_id=full_id,
            status="running",
            service_url=f"http://localhost:{request.port}"
        )
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/push_to_hub")
def push_to_hub(request: PushRequest, token: Optional[str] = Depends(get_token)):
    """
    Push a trained model/adapter to Hugging Face Hub.
    """
    if not token:
        raise HTTPException(status_code=401, detail="Hugging Face token required to push to Hub.")

    try:
        msg = job_manager.push_to_hub(
            job_id=request.job_id,
            hub_model_id=request.hub_model_id,
            private=request.private,
            hf_token=token
        )
        return {"status": "success", "message": msg}
    except Exception as e:
        logger.error(f"Push to hub failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/train/{job_id}/logs")
async def websocket_logs(websocket: WebSocket, job_id: str):
    """
    Stream logs for a training or serving job.
    """
    await websocket.accept()
    try:
        async for line in job_manager.stream_logs(job_id):
            await websocket.send_text(line)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
