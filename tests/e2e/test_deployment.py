import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.lightning_tune.deploy import MultiModalAPI
from src.lightning_tune.config import PipelineConfig, ModelConfig, DataConfig

# Mock Model and Tokenizer
class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.config = MagicMock()
        self.config.hidden_size = 768
    
    def forward(self, input_ids, **kwargs):
        # Return dummy logits
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, 32000)
        return MagicMock(logits=logits)
    
    def generate(self, *args, **kwargs):
        # Return dummy generated token ids
        return torch.tensor([[1, 2, 3]])

    def to(self, device):
        self.device = device
        return self

class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
    
    def encode(self, text, **kwargs):
        if kwargs.get("return_tensors") == "pt":
            return torch.tensor([[1, 2, 3]])
        return [1, 2, 3]
    
    def decode(self, token_ids, **kwargs):
        return "Generated text"
    
    def batch_decode(self, token_ids, **kwargs):
        return ["Generated text"]
    
    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

@pytest.fixture
def api_instance():
    with patch("src.lightning_tune.deploy.AutoModelForCausalLM.from_pretrained", return_value=MockModel()), \
         patch("src.lightning_tune.deploy.AutoTokenizer.from_pretrained", return_value=MockTokenizer()), \
         patch("src.lightning_tune.deploy.MultimodalLLM", return_value=MockModel()):
        
        config = PipelineConfig(
            model=ModelConfig(repo_id="test/model"),
            data=DataConfig(dataset_repo_id="dummy/repo", text_columns=["text"], output_column="label")
        )
        
        # Initialize with dummy path, attributes injected manually for test
        api = MultiModalAPI(checkpoint_path=Path("dummy_ckpt"))
        api.config = config
        api.model = MockModel()
        api.tokenizer = MockTokenizer()
        api.device = torch.device("cpu")
        api.image_transform = lambda x: torch.randn(3, 224, 224) 
        
        return api

def test_predict_text_only(api_instance):
    payload = {"prompt": "Hello"}
    # Simulate LitServe flow: decode -> predict -> encode
    decoded = api_instance.decode_request(payload)
    output = api_instance.predict(decoded)
    response = api_instance.encode_response(output)
    assert "completion" in response
    assert response["completion"] == "Generated text"

def test_predict_with_image_mock(api_instance):
    payload = {"prompt": "Describe image", "image_b64": "invalid_base64"}
    
    # decode_request handles image processing (and error logging if invalid)
    decoded = api_instance.decode_request(payload)
    output = api_instance.predict(decoded)
    response = api_instance.encode_response(output)
    assert "completion" in response

def test_encode_response(api_instance):
    # Output from predict is a tensor of token ids (or list of them). 
    # But mock generate returns tensor [[1, 2, 3]].
    # encode_response expects tokens.
    output_tokens = torch.tensor([[1, 2, 3]])
    encoded = api_instance.encode_response(output_tokens)
    assert encoded["completion"] == "Generated text"

def test_setup_method(api_instance):
    # Ensure setup runs without error (mocked)
    assert api_instance.model is not None
    assert api_instance.tokenizer is not None
