import torch, base64, io
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


class TextLLMAPI(LitAPI):
    def __init__(self, adapter_path: Path, config: PipelineConfig):
        self.adapter_path = adapter_path
        self.config = config

    def setup(self, device):
        # Determine model class
        hf_config = AutoConfig.from_pretrained(self.config.model.repo_id, trust_remote_code=True)
        # Check if vision model (Native VLM used as text generator?)
        # Or just standard CausalLM.
        # For robustness, try AutoModelForCausalLM first as SFTTrainer uses it.
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model.repo_id,
                return_dict=True,
                torch_dtype=torch.bfloat16,
                cache_dir=self.config.model.base_model_dir,
                trust_remote_code=True
            )
        except Exception:
             # Fallback for some models
             base_model = AutoModelForVision2Seq.from_pretrained(
                self.config.model.repo_id,
                return_dict=True,
                torch_dtype=torch.bfloat16,
                cache_dir=self.config.model.base_model_dir,
                trust_remote_code=True
            )

        self.model = (
            PeftModel.from_pretrained(base_model, str(self.adapter_path))
            .merge_and_unload()
            .to(device)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.repo_id,
            cache_dir=self.config.model.base_model_dir,
            trust_remote_code=True
        )

    def decode_request(self, r: dict):
        # Support simple chat format or raw prompt
        prompt = r.get("prompt", "")
        # If messages list is passed (chat), apply template?
        # For now assume raw prompt or simple text.
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)

    @torch.inference_mode()
    def predict(self, x):
        outputs = self.model.generate(input_ids=x["input_ids"], max_new_tokens=100)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    def encode_response(self, text) -> dict:
        return {"completion": text}


class VLLMAPI(LitAPI):
    def __init__(self, adapter_path: Path, config: PipelineConfig):
        self.adapter_path = adapter_path
        self.config = config
        self.lora_request = None

    def setup(self, device):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. Please install it with `pip install vllm`.")
        
        # Initialize vLLM
        # Enable LoRA if adapter is present
        self.llm_engine = LLM(
            model=str(self.config.model.repo_id),
            enable_lora=True,
            max_lora_rank=64, # Default max
            trust_remote_code=True,
            gpu_memory_utilization=0.8
            # device is handled by vLLM (uses CUDA_VISIBLE_DEVICES or internal logic)
        )
        
        # Define LoRA request
        # We assign a unique ID (e.g., 1) and give it a name
        adapter_name = self.adapter_path.name
        self.lora_request = LoRARequest(adapter_name, 1, str(self.adapter_path))
        
        self.sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

    def decode_request(self, r: dict):
        return r.get("prompt", "")

    def predict(self, x):
        # x is a list of prompts (batch) from LitServe if batched, or single
        # LitServe usually passes batched inputs to predict if batching is enabled.
        # But here let's assume simple handling.
        
        # vLLM generate expects list of prompts
        prompts = x if isinstance(x, list) else [x]
        
        outputs = self.llm_engine.generate(
            prompts,
            self.sampling_params,
            lora_request=self.lora_request
        )
        return [o.outputs[0].text for o in outputs]

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
        self.checkpoint_path = checkpoint_path

    def setup(self, device):
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
            ).to(self.device)
        }
        # Image
        if "image_b64" in request and self.config.data.vision_config:
            try:
                img_bytes = base64.b64decode(request["image_b64"])
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                # Apply transform
                item["image"] = self.image_transform(img).unsqueeze(0).to(self.device)
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
                 
                 item["audio"] = waveform.mean(0).unsqueeze(0).to(self.device) # [1, T]
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
                    dtype=torch.float32,
                    device=self.device,
                )
            if tc.categorical_columns:
                item["tabular_cat"] = torch.tensor(
                    [
                        [
                            self.preprocessors["cat_mappings"][c][request["tabular"][c]]
                            for c in tc.categorical_columns
                        ]
                    ],
                    dtype=torch.long,
                    device=self.device,
                )

        # Labels needed for forward? No, generate doesn't need labels.
        return item

    @torch.inference_mode()
    def predict(self, batch):
        return self.model.generate(batch, self.tokenizer, max_new_tokens=100)

    def encode_response(self, tokens) -> dict:
        text = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
        return {"completion": text}


def launch_server(config: PipelineConfig, trained_artifact_path: Path):
    if config.is_multimodal:
        api = MultiModalAPI(checkpoint_path=trained_artifact_path)
    else:
        # Check if vLLM requested or available
        # logic: if generic LLM, prefer vLLM if installed AND requested (default True)
        if VLLM_AVAILABLE and config.deployment.use_vllm:
             print("⚡ Using vLLM for high-performance serving ⚡")
             api = VLLMAPI(adapter_path=trained_artifact_path, config=config)
        else:
             api = TextLLMAPI(adapter_path=trained_artifact_path, config=config)

    if config.trainer.device == "cuda":
        api.model = torch.compile(api.model)

    server = LitServer(api, accelerator="auto", devices=1)
    print(f"\n🚀 Server launching on http://127.0.0.1:{config.deployment.port} 🚀\n")
    server.run(port=config.deployment.port)
