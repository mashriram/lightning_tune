from fastapi import FastAPI, HTTPException, Header, Depends, WebSocket, UploadFile, File
from typing import List, Optional, Union, Dict, Any
from pathlib import Path
import shutil
import uuid
from .schemas import SearchResult, DatasetSearchResult, AnalyzeRequest, TrainRequest, JobResponse, ServeRequest, PushRequest
from .job_manager import job_manager
from src.lightning_tune.hf_utils import search_models, search_datasets
from src.lightning_tune.config import PipelineConfig
import logging

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
        kwargs = {}
        if request.file_path:
            kwargs["file_path"] = Path(request.file_path)
            # If file path implies multimodal (images in zip?), usually handled by user ensuring paths are relative?
            # Or simplified: we assume images are not uploaded via single file endpoint easily for now unless zip.
            # But PipelineConfig usually expects a csv pointing to images on disk.
            # For simplicity in this demo API, we treat upload as the data file.
            # Image root path might be an issue if images are separate.
            # We assume user uploads self-contained or text-only if single file.
            # If zip, we could extract.
            # For now, pass file_path.
            if request.file_path.endswith(".zip"):
                 # Optional: Unzip logic?
                 pass
        elif request.dataset_repo_id:
            kwargs["dataset_repo_id"] = request.dataset_repo_id
            kwargs["split"] = request.split
        else:
            raise HTTPException(status_code=400, detail="Provide either dataset_repo_id or file_path")

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
