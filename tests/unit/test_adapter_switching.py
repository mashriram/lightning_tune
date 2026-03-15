import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.lightning_tune.deploy import TextLLMAPI, VLLMAPI, VLLM_AVAILABLE
from src.lightning_tune.config import PipelineConfig, ModelConfig, DataConfig

@pytest.fixture
def mock_peft_model():
    model = MagicMock()
    model.active_adapter = "default"
    model.peft_config = {"default": MagicMock()}
    return model

@pytest.fixture
def text_api(mock_peft_model):
    config = PipelineConfig(
        model=ModelConfig(repo_id="tiny-llm"),
        data=DataConfig(file_path="dummy.csv")
    )
    api = TextLLMAPI(base_adapter_path=Path("base_adapter"), config=config)
    
    # Manually inject mocks to avoid full setup
    api.model = mock_peft_model
    api.tokenizer = MagicMock()
    api.tokenizer.decode.return_value = "generated response"
    api.tokenizer.batch_decode.return_value = ["generated response"]
    api.device = "cpu"
    api.adapters_cached = {"default": "base_adapter"}
    
    return api

def test_text_llm_adapter_switching(text_api):
    # 1. Test request without adapter_path (uses default)
    request = {"prompt": "hello"}
    decoded = text_api.decode_request(request)
    text_api.predict(decoded)
    
    assert text_api.model.set_adapter.call_count == 0
    
    # 2. Test request with a NEW adapter_path
    new_adapter_path = "/path/to/new_adapter"
    request = {"prompt": "hello", "adapter_path": new_adapter_path}
    decoded = text_api.decode_request(request)
    
    # Mock return for generate
    text_api.model.generate.return_value = torch.tensor([[1, 2, 3]])
    
    text_api.predict(decoded)
    
    # Should have called load_adapter and set_adapter
    text_api.model.load_adapter.assert_called_once_with(new_adapter_path, adapter_name="new_adapter")
    text_api.model.set_adapter.assert_called_once_with("new_adapter")
    assert text_api.adapters_cached["new_adapter"] == new_adapter_path

def test_text_llm_cached_adapter_switching(text_api):
    # Pre-cache an adapter
    text_api.adapters_cached["extra"] = "/path/to/extra"
    text_api.model.peft_config["extra"] = MagicMock()
    
    request = {"prompt": "hello", "adapter_path": "/path/to/extra"}
    decoded = text_api.decode_request(request)
    
    text_api.predict(decoded)
    
    # Should NOT call load_adapter (already cached), but should call set_adapter
    assert text_api.model.load_adapter.call_count == 0
    text_api.model.set_adapter.assert_called_once_with("extra")

@patch("src.lightning_tune.deploy.LoRARequest")
def test_vllm_adapter_switching(mock_lora_cls):
    config = PipelineConfig(
        model=ModelConfig(repo_id="tiny-llm"),
        data=DataConfig(file_path="dummy.csv")
    )
    api = VLLMAPI(base_adapter_path=Path("base_adapter"), config=config)
    
    # Mock LLM engine
    mock_llm = MagicMock()
    api.llm_engine = mock_llm
    api.lora_requests = {"default": MagicMock()}
    api.sampling_params = MagicMock()
    
    # Mock the LoRARequest return instance
    mock_lora_instance = MagicMock()
    mock_lora_cls.return_value = mock_lora_instance
    
    # 1. Request with new adapter
    new_adapter_path = "/path/to/vllm_adapter"
    request = {"prompt": "hello", "adapter_path": new_adapter_path}
    decoded = api.decode_request(request)
    
    api.predict(decoded)
    
    # Should have created a new LoRARequest
    assert "vllm_adapter" in api.lora_requests
    new_req = api.lora_requests["vllm_adapter"]
    assert new_req == mock_lora_instance
    
    # Check generate call
    mock_llm.generate.assert_called_once()
    args, kwargs = mock_llm.generate.call_args
    assert kwargs["lora_request"] == new_req
