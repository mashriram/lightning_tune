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
    dataset_repo_id: Optional[str] = None
    file_path: Optional[str] = None
    db_uri: Optional[str] = None
    db_query: Optional[str] = None
    split: Optional[str] = None
    datasets: Optional[List[Dict[str, Any]]] = None

class TrainRequest(BaseModel):
    config: Dict[str, Any]
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None

class ServeRequest(BaseModel):
    job_id: Optional[str] = None
    model_path: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    port: int = 8000
    use_vllm: bool = True

class JobResponse(BaseModel):
    job_id: str
    status: str
    output_dir: Optional[str] = None
    service_url: Optional[str] = None

class PushRequest(BaseModel):
    job_id: str
    hub_model_id: str
    private: bool = False
