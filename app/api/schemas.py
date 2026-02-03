from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class SearchResult(BaseModel):
    id: str
    downloads: int
    likes: int
    private: bool
    task: Optional[str] = None

class DatasetSearchResult(BaseModel):
    id: str
    downloads: int
    likes: int
    private: bool

class AnalyzeRequest(BaseModel):
    model_repo_id: str
    dataset_repo_id: str
    split: Optional[str] = None

class TrainRequest(BaseModel):
    config: Dict[str, Any]
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None

class ServeRequest(BaseModel):
    job_id: Optional[str] = None
    model_path: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    port: int = 8000

class JobResponse(BaseModel):
    job_id: str
    status: str
    output_dir: Optional[str] = None
    service_url: Optional[str] = None

class PushRequest(BaseModel):
    job_id: str
    hub_model_id: str
    private: bool = False
