from fastapi import FastAPI, HTTPException, Header, Depends, WebSocket
from typing import List, Optional
from .schemas import SearchResult, DatasetSearchResult, AnalyzeRequest, TrainRequest, JobResponse, ServeRequest
from .job_manager import job_manager
from src.lightning_tune.hf_utils import search_models, search_datasets
from src.lightning_tune.config import PipelineConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(title="Lightning Tune API")

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

@app.post("/analyze")
def analyze_dataset(request: AnalyzeRequest, token: Optional[str] = Depends(get_token)):
    """
    Analyze a HF dataset and suggest configuration.
    """
    try:
        config = PipelineConfig.from_dataset(
            model_repo_id=request.model_repo_id,
            dataset_repo_id=request.dataset_repo_id,
            token=token
        )
        return config.model_dump()
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
        # Use a prefixed ID to distinguish service jobs
        full_id = f"service_{service_id}"
        return JobResponse(
            job_id=full_id,
            status="running",
            service_url=f"http://localhost:{request.port}"
        )
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
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
