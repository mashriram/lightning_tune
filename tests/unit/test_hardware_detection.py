from lightning_tune.utils import get_auto_device
import torch

def test_get_auto_device():
    if torch.cuda.is_available():
        assert get_auto_device() == "cuda"
    elif torch.backends.mps.is_available():
        assert get_auto_device() == "mps"
    else:
        assert get_auto_device() == "cpu"
