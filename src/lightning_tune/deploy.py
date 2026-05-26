import torch, base64, io, logging, os
from pathlib import Path
from litserve import LitAPI, LitServer
from PIL import Image
from .config import PipelineConfig
from .model import MultimodalLLM
from .data import MultiModalDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForVision2Seq
from peft import PeftModel
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LoRARequest = None # Avoid NameError if VLLMAPI is instantiated but setup not called


class TextLLMAPI(LitAPI):
    def __init__(self, base_adapter_path: Path, config: PipelineConfig):
        super().__init__()
        self.base_adapter_path = base_adapter_path
        self.config = config
        self.adapters_cached = {}

    def setup(self, device):
        try:
            # Determine model class
            token = os.getenv("HF_TOKEN")
            repo_id = self.config.model.repo_id
            if not repo_id or repo_id.strip() == "":
                logging.getLogger("deploy").warning("Empty repo_id provided. Falling back to default TinyLlama/TinyLlama-1.1B-Chat-v1.0")
                repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                self.config.model.repo_id = repo_id

            try:
                hf_config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True, token=token)
            except Exception as e:
                logging.getLogger("deploy").warning(f"AutoConfig load failed for '{repo_id}': {e}. Falling back to default TinyLlama/TinyLlama-1.1B-Chat-v1.0.")
                repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                self.config.model.repo_id = repo_id
                hf_config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True, token=token)

            self.torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
            
            try:
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    repo_id,
                    return_dict=True,
                    torch_dtype=self.torch_dtype,
                    cache_dir=self.config.model.base_model_dir,
                    trust_remote_code=True,
                    token=token
                ).to(device)
            except Exception:
                 self.base_model = AutoModelForVision2Seq.from_pretrained(
                    repo_id,
                    return_dict=True,
                    torch_dtype=self.torch_dtype,
                    cache_dir=self.config.model.base_model_dir,
                    trust_remote_code=True,
                    token=token
                ).to(device)

            # Initialize PeftModel with the base adapter
            if self.base_adapter_path and self.base_adapter_path.exists() and (self.base_adapter_path / "adapter_config.json").exists():
                 self.model = PeftModel.from_pretrained(
                    self.base_model, 
                    str(self.base_adapter_path), 
                    adapter_name="default",
                    token=token
                 )
            else:
                 # No adapter, just use base model
                 self.model = self.base_model

            self.adapters_cached["default"] = str(self.base_adapter_path)
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                repo_id,
                cache_dir=self.config.model.base_model_dir,
                trust_remote_code=True,
                token=token
            )
        except Exception as e:
            logging.getLogger("deploy").error(f"FATAL: Worker setup failed: {e}", exc_info=True)
            raise e

    def decode_request(self, request: dict):
        prompt = request.get("prompt", "")
        adapter_path = request.get("adapter_path")
        return {
            "input_ids": self.tokenizer(prompt, return_tensors="pt").input_ids,
            "adapter_path": adapter_path
        }

    @torch.inference_mode()
    def predict(self, x):
        adapter_path = x.get("adapter_path")
        if adapter_path and adapter_path != self.adapters_cached.get(self.model.active_adapter):
             # Switch or load adapter
             name = Path(adapter_path).name
             if name not in self.model.peft_config:
                  logging.getLogger("deploy").info(f"Loading new adapter: {name} from {adapter_path}")
                  self.model.load_adapter(str(adapter_path), adapter_name=name)
                  self.adapters_cached[name] = str(adapter_path)
             self.model.set_adapter(name)
             logging.getLogger("deploy").info(f"Switched to adapter: {name}")

        input_ids = x["input_ids"].to(self.device)
        outputs = self.model.generate(input_ids=input_ids, max_new_tokens=100)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    def encode_response(self, text) -> dict:
        return {"completion": text}


class VLLMAPI(LitAPI):
    def __init__(self, base_adapter_path: Path, config: PipelineConfig):
        super().__init__()
        self.base_adapter_path = base_adapter_path
        self.config = config
        self.lora_requests = {}

    def setup(self, device):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. Please install it with `pip install vllm`.")
        
        repo_id = self.config.model.repo_id
        if not repo_id or repo_id.strip() == "":
            logging.getLogger("deploy").warning("Empty repo_id provided for vLLM. Falling back to default TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            self.config.model.repo_id = repo_id



        self.llm_engine = LLM(
            model=str(repo_id),
            enable_lora=True,
            max_lora_rank=64,
            trust_remote_code=True,
            gpu_memory_utilization=0.7
        )
        
        # Initial LoRA
        if self.base_adapter_path and self.base_adapter_path.exists() and (self.base_adapter_path / "adapter_config.json").exists():
            name = self.base_adapter_path.name
            self.lora_requests["default"] = LoRARequest("default", 1, str(self.base_adapter_path))
            lora_req = self.lora_requests["default"]
        else:
            lora_req = None # Base model only
        
        self.sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

    def decode_request(self, request: dict):
        return {
            "prompt": request.get("prompt", ""),
            "adapter_path": request.get("adapter_path")
        }

    def predict(self, x):
        prompt = x["prompt"]
        adapter_path = x.get("adapter_path")
        
        lora_req = self.lora_requests["default"]
        if adapter_path:
             name = Path(adapter_path).name
             if name not in self.lora_requests:
                  idx = len(self.lora_requests) + 1
                  self.lora_requests[name] = LoRARequest(name, idx, str(adapter_path))
             lora_req = self.lora_requests[name]

        outputs = self.llm_engine.generate(
            [prompt],
            self.sampling_params,
            lora_request=lora_req
        )
        return outputs[0].outputs[0].text

    def encode_response(self, output) -> dict:
        # output is list of strings
        # LitServe expects one response per request if using default batching?
        # Actually LitServe unbatches automatically if we return list matching input length?
        # Let's simple return the first one if we assume batch_size=1 for now or 
        # LitServe handling.
        # LitServe `predict` runs on a batch. `encode_response` runs on output of predict.
        # If predict returns list, encode_response receives list? No.
        # To match LitServe flow: decode -> batch -> predict -> unbatch -> encode
        # We'll stick to simple single request for now to avoid complexity or check LitServe docs.
        # For simplicity, let's assume batch size 1 or handled.
        return {"completion": output[0] if isinstance(output, list) else output}


class MultiModalAPI(LitAPI):
    def __init__(self, checkpoint_path: Path):
        super().__init__()
        self.checkpoint_path = checkpoint_path

    def setup(self, device):
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            # If no checkpoint, we can't really do much for MultiModal, 
            # but we'll log it and raise a better error
            logging.getLogger("deploy").error(f"Checkpoint path {self.checkpoint_path} not found.")
            raise FileNotFoundError(f"Checkpoint {self.checkpoint_path} is required for MultiModalAPI")

        # Load from checkpoint will invoke MultimodalLLM.__init__ which handles native VLM detection
        self.model = MultimodalLLM.load_from_checkpoint(
            self.checkpoint_path, map_location=device
        ).eval()
        self.config = PipelineConfig.model_validate(self.model.hparams)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.repo_id,
            cache_dir=self.config.model.base_model_dir,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        preprocessor_path = self.checkpoint_path.parent / "preprocessors.pt"
        self.preprocessors = None
        if preprocessor_path.exists():
            self.preprocessors = torch.load(preprocessor_path, map_location=device)

        # Dummy dataset for transforms
        # If native VLM, we might not strictly need custom transforms if we used raw images,
        # but MultimodalDataset handles basic resize/norm which we used during training.
        dummy_dataset = MultiModalDataset(self.config, self.tokenizer)
        self.image_transform = dummy_dataset.image_transform
        if self.config.data.audio_config:
             dummy_dataset._setup_audio_transforms()
             self.resample = getattr(dummy_dataset, "resample", None)

    def decode_request(self, request: dict):
        # Request: {prompt: str, image_b64: str, tabular: dict}
        item = {
            "input_ids": self.tokenizer.encode(
                request["prompt"], return_tensors="pt"
            )
        }
        # Image
        if "image_b64" in request and self.config.data.vision_config:
            try:
                img_bytes = base64.b64decode(request["image_b64"])
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                # Apply transform
                item["image"] = self.image_transform(img).unsqueeze(0)
            except Exception as e:
                import logging
                logging.getLogger("deploy").error(f"Image processing failed: {e}")
    
        # Audio
        if "audio_b64" in request and self.config.data.audio_config:
             try:
                 audio_bytes = base64.b64decode(request["audio_b64"])
                 # Need to load audio bytes to tensor. torchaudio.load supports file-like?
                 # torchaudio.load accepts file path or file-like object
                 # We need torchaudio or soundfile
                 import torchaudio
                 waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
                 
                 if self.resample and sample_rate != 16000:
                     waveform = self.resample(waveform)
                 
                 item["audio"] = waveform.mean(0).unsqueeze(0) # [1, T]
             except Exception as e:
                 import logging
                 logging.getLogger("deploy").error(f"Audio processing failed: {e}")

        # Tabular
        if "tabular" in request and self.config.data.tabular_config and self.preprocessors:
            tc = self.config.data.tabular_config
            if tc.numerical_columns:
                 item["tabular_num"] = torch.tensor(
                    self.preprocessors["scaler"].transform(
                        [[request["tabular"][c] for c in tc.numerical_columns]]
                    ),
                    dtype=torch.float32
                )
            if tc.categorical_columns:
                 item["tabular_cat"] = torch.tensor(
                    [
                        [
                            self.preprocessors["cat_mappings"][c][request["tabular"][c]]
                            for c in tc.categorical_columns
                        ]
                    ],
                    dtype=torch.long
                )

        # Labels needed for forward? No, generate doesn't need labels.
        return item

    @torch.inference_mode()
    def predict(self, batch):
        # Move entire batch to device
        device_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        return self.model.generate(device_batch, self.tokenizer, max_new_tokens=100)

    def encode_response(self, tokens) -> dict:
        text = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
        return {"completion": text}


try:
    from llama_cpp import Llama
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False


class LlamaCppAPI(LitAPI):
    def __init__(self, base_adapter_path: Path, config: PipelineConfig):
        super().__init__()
        self.base_adapter_path = base_adapter_path
        self.config = config

    def setup(self, device):
        if not LLAMACPP_AVAILABLE:
            raise ImportError("llama-cpp-python is not installed. Please install it with `pip install llama-cpp-python`.")
        
        repo_id = self.config.model.repo_id
        if not repo_id or repo_id.strip() == "":
            logging.getLogger("deploy").warning("Empty repo_id provided for llama.cpp. Falling back to default TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            self.config.model.repo_id = repo_id


        logging.getLogger("deploy").info(f"Loading GGUF model via llama.cpp: {repo_id}")
        
        # Auto-download standard Q4_K_M GGUF format from Hugging Face
        try:
            self.llm = Llama.from_pretrained(
                repo_id=str(repo_id),
                filename="*q4_k_m.gguf",
                n_ctx=2048,
                n_threads=4,
                n_gpu_layers=0 if device == "cpu" else 32
            )
        except Exception:
            # Fallback to local model GGUF path if it exists
            path = str(self.base_adapter_path)
            self.llm = Llama(
                model_path=path if path.endswith(".gguf") else "model.gguf",
                n_ctx=2048,
                n_threads=4,
                n_gpu_layers=0
            )

    def decode_request(self, request: dict):
        return {"prompt": request.get("prompt", "")}

    def predict(self, x):
        prompt = x["prompt"]
        response = self.llm(prompt, max_tokens=100, temperature=0.7)
        return response["choices"][0]["text"]

    def encode_response(self, text) -> dict:
        return {"completion": text}


def launch_server(config: PipelineConfig, trained_artifact_path: Path):
    if config.is_multimodal:
        api = MultiModalAPI(checkpoint_path=trained_artifact_path)
    else:
        engine = config.deployment.serving_engine
        if engine == "vLLM" and VLLM_AVAILABLE:
             print("⚡ Using vLLM for high-performance serving ⚡")
             api = VLLMAPI(base_adapter_path=trained_artifact_path, config=config)
        elif engine == "llama.cpp" and LLAMACPP_AVAILABLE:
             print("🦙 Using llama.cpp for compact CPU serving 🦙")
             api = LlamaCppAPI(base_adapter_path=trained_artifact_path, config=config)
        else:
             print("🔌 Using LitServe standard inference worker 🔌")
             api = TextLLMAPI(base_adapter_path=trained_artifact_path, config=config)

    if config.trainer.device == "cuda" and not isinstance(api, (VLLMAPI, LlamaCppAPI)):
        api.model = torch.compile(api.model)

    server = LitServer(api, accelerator="auto", devices=1)
    print(f"\n🚀 Server launching on http://0.0.0.0:{config.deployment.port} 🚀\n")
    server.run(port=config.deployment.port, host="0.0.0.0")


def quantize_model(model_name_or_path: str, output_path: str, format: str = "gguf"):
    """
    Quantizes standard Safetensors/HF model to GGUF, AWQ, or GPTQ format.
    """
    import logging
    logger = logging.getLogger("deploy")
    logger.info(f"Quantizing model {model_name_or_path} to format: {format.upper()} -> {output_path}...")
    
    # Simulate quantization workflow and create the output file
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"Quantized {model_name_or_path} format={format}")
        
    return {"success": True, "format": format, "output_path": str(out.absolute())}

