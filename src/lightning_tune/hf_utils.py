from huggingface_hub import HfApi, utils
from typing import List, Optional, Dict, Any
from datasets import get_dataset_split_names
import logging

logger = logging.getLogger(__name__)

def search_models(query: str, limit: int = 20, token: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Search for models on Hugging Face Hub.
    """
    api = HfApi(token=token)
    try:
        models = api.list_models(search=query, limit=limit, sort="downloads", direction=-1)
        return [
            {
                "id": m.modelId,
                "downloads": m.downloads,
                "likes": m.likes,
                "task": m.pipeline_tag,
                "private": m.private,
            }
            for m in models
        ]
    except utils.RepositoryNotFoundError:
        return []
    except Exception as e:
        logger.error(f"Error searching models: {e}")
        raise

def search_datasets(query: str, limit: int = 20, token: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Search for datasets on Hugging Face Hub.
    """
    api = HfApi(token=token)
    try:
        datasets = api.list_datasets(search=query, limit=limit, sort="downloads", direction=-1)
        return [
            {
                "id": d.id,
                "downloads": d.downloads,
                "likes": d.likes,
                "private": d.private,
            }
            for d in datasets
        ]
    except Exception as e:
        logger.error(f"Error searching datasets: {e}")
        raise

def get_dataset_info(repo_id: str, token: Optional[str] = None) -> Dict[str, Any]:
    """
    Get information about a specific dataset.
    Raises an exception if not found or unauthorized.
    """
    api = HfApi(token=token)
    info = api.dataset_info(repo_id)
    return {
        "id": info.id,
        "private": info.private,
        "downloads": info.downloads,
        "likes": info.likes,
        "tags": info.tags,
    }

def get_dataset_splits(repo_id: str, token: Optional[str] = None) -> List[str]:
    """
    Get available splits for a dataset.
    """
    try:
        # get_dataset_split_names is efficient and doesn't download data
        return get_dataset_split_names(repo_id, token=token)
    except Exception as e:
        logger.error(f"Error getting splits for {repo_id}: {e}")
        # Fallback or re-raise? Re-raising lets caller handle it (e.g. auth error)
        raise
