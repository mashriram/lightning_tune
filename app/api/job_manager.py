import subprocess
import uuid
import os
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, AsyncGenerator
import asyncio
import logging
import yaml
from huggingface_hub import HfApi

JOBS_DIR = Path("jobs")
JOBS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("job_manager")

class JobManager:
    def __init__(self):
        self.active_jobs: Dict[str, subprocess.Popen] = {}

    def start_training_job(self, config_dict: Dict[str, Any], hf_token: Optional[str] = None) -> str:
        job_id = str(uuid.uuid4())
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = job_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        log_path = job_dir / "train.log"
        log_file = open(log_path, "w")

        env = os.environ.copy()
        if hf_token:
            env["HF_TOKEN"] = hf_token
            env["HUGGING_FACE_HUB_TOKEN"] = hf_token

        cmd = [sys.executable, "-m", "lightning_tune.cli", "train", str(config_path)]

        logger.info(f"Starting job {job_id} with cmd: {cmd}")

        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=os.getcwd()
        )

        # Close file handle in parent process
        log_file.close()

        self.active_jobs[job_id] = process
        return job_id

    def start_tuning_job(self, config_dict: Dict[str, Any], n_trials: int = 10, hf_token: Optional[str] = None) -> str:
        job_id = str(uuid.uuid4())
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        config_path = job_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        log_path = job_dir / "tune.log"
        log_file = open(log_path, "w")

        env = os.environ.copy()
        if hf_token:
            env["HF_TOKEN"] = hf_token
            env["HUGGING_FACE_HUB_TOKEN"] = hf_token

        cmd = [sys.executable, "-m", "lightning_tune.cli", "tune", str(config_path), "--n-trials", str(n_trials)]

        logger.info(f"Starting tuning job {job_id} with cmd: {cmd}")

        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=os.getcwd()
        )
        log_file.close()

        self.active_jobs[job_id] = process
        return job_id

    def start_serving_job(self, job_id: Optional[str], model_path: Optional[str], config: Optional[Dict], port: int, use_vllm: bool, hf_token: Optional[str]) -> str:
        service_id = str(uuid.uuid4())
        service_dir = JOBS_DIR / f"service_{service_id}"
        service_dir.mkdir(parents=True, exist_ok=True)

        cfg_path = None
        mdl_path = None

        if job_id:
            job_dir = JOBS_DIR / job_id
            if not job_dir.exists():
                raise ValueError(f"Job {job_id} not found")

            if (job_dir / "config.yaml").exists():
                cfg_path = job_dir / "config.yaml"

            # Try to find model artifact
            candidates = ["final_adapter", "final.ckpt", "best_model.ckpt"]
            for c in candidates:
                if (job_dir / c).exists():
                    mdl_path = job_dir / c
                    break

        if model_path:
            mdl_path = Path(model_path)

        if config is not None:
            base_cfg = {}
            if cfg_path:
                with open(cfg_path, "r") as f:
                    base_cfg = yaml.safe_load(f)

            base_cfg.update(config)

            cfg_path = service_dir / "user_config.yaml"
            with open(cfg_path, "w") as f:
                yaml.dump(base_cfg, f)

        if not cfg_path or not cfg_path.exists():
            cfg_path = service_dir / "user_config.yaml"
            with open(cfg_path, "w") as f:
                yaml.dump({"model": {"name": str(mdl_path)}}, f)

        is_hub_id = mdl_path and "/" in str(mdl_path)
        if not mdl_path or (not mdl_path.exists() and not is_hub_id):
            raise ValueError("Model artifact not found locally and does not resemble a valid Hub ID. Please provide model_path or valid job_id with artifacts.")

        # Update port in config
        with open(cfg_path, "r") as f:
            final_cfg = yaml.safe_load(f) or {}

        if "model" not in final_cfg:
            final_cfg["model"] = {"repo_id": str(mdl_path)}
        elif "name" in final_cfg["model"]:
            final_cfg["model"]["repo_id"] = final_cfg["model"].pop("name")
            
        if "data" not in final_cfg:
            final_cfg["data"] = {"dataset_repo_id": "dummy/minimal"}

        if "deployment" not in final_cfg:
            final_cfg["deployment"] = {}
        final_cfg["deployment"]["port"] = port
        final_cfg["deployment"]["use_vllm"] = use_vllm

        final_cfg_path = service_dir / "serve_config.yaml"
        with open(final_cfg_path, "w") as f:
            yaml.dump(final_cfg, f)

        log_path = service_dir / "serve.log"
        log_file = open(log_path, "w")

        env = os.environ.copy()
        if hf_token:
            env["HF_TOKEN"] = hf_token
            env["HUGGING_FACE_HUB_TOKEN"] = hf_token

        cmd = [sys.executable, "-m", "lightning_tune.cli", "serve", str(final_cfg_path), str(mdl_path)]

        logger.info(f"Starting service {service_id} on port {port}")

        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=os.getcwd()
        )

        # Close file handle in parent process
        log_file.close()

        self.active_jobs[f"service_{service_id}"] = process
        return service_id

    def stop_job(self, job_id: str) -> bool:
        if job_id in self.active_jobs:
            process = self.active_jobs[job_id]
            if process.poll() is None:
                 process.terminate()
                 try:
                     process.wait(timeout=5)
                 except subprocess.TimeoutExpired:
                     process.kill()
                 return True
        return False

    def push_to_hub(self, job_id: str, hub_model_id: str, private: bool, hf_token: str):
        job_dir = JOBS_DIR / job_id
        if not job_dir.exists():
            raise ValueError(f"Job {job_id} not found")

        # Find artifacts
        artifact_path = None
        if (job_dir / "final_adapter").exists():
            # Text pipeline adapter
            artifact_path = job_dir / "final_adapter"
            # It's a folder, upload folder
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=hub_model_id, private=private, exist_ok=True)
            api.upload_folder(folder_path=str(artifact_path), repo_id=hub_model_id, repo_type="model")
            return f"Uploaded adapter to {hub_model_id}"

        elif (job_dir / "final.ckpt").exists() or (job_dir / "best_model.ckpt").exists():
            # Multimodal pipeline
            # Just upload the whole job dir (excluding logs maybe) or just checkpoints?
            # User usually wants the checkpoint.
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=hub_model_id, private=private, exist_ok=True)
            api.upload_folder(folder_path=str(job_dir), repo_id=hub_model_id, repo_type="model", ignore_patterns=["*.log", "config.yaml"])
            return f"Uploaded checkpoints to {hub_model_id}"

        else:
            raise ValueError("No artifacts found to push.")

    def get_job_status(self, job_id: str) -> str:
        if job_id not in self.active_jobs:
            if (JOBS_DIR / job_id).exists():
                 return "stopped"
            if (JOBS_DIR / f"service_{job_id}").exists(): # Handle service prefix in ID if user passed just UUID
                 return "stopped"
            return "not_found"

        process = self.active_jobs[job_id]
        ret = process.poll()
        if ret is None:
            return "running"
        elif ret == 0:
            return "completed"
        else:
            return "failed"

    async def stream_logs(self, job_id: str) -> AsyncGenerator[str, None]:
        # Handle service logs too?
        # If job_id starts with service_, look in service dir
        if job_id.startswith("service_"):
             log_path = JOBS_DIR / job_id / "serve.log"
        else:
             log_path = JOBS_DIR / job_id / "train.log"

        if not log_path.exists():
            yield "Log file not found."
            return

        with open(log_path, "r") as f:
            while True:
                line = f.readline()
                if line:
                    yield line
                else:
                    status = self.get_job_status(job_id)
                    if status != "running":
                        yield f.read()
                        break
                    await asyncio.sleep(0.5)

job_manager = JobManager()
