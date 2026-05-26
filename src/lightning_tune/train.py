import lightning as L, torch, math, warnings, logging, os
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from functools import partial
from .config import PipelineConfig
from .data import prepare_text_dataset, MultiModalDataset, StreamingMultiModalDataset, multimodal_collate_fn
from .model import MultimodalLLM
from .hf_utils import get_dataset_splits
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoConfig
)
from typing import Dict, Union, Any, Iterator, List, Optional
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig, GRPOTrainer, GRPOConfig
from transformers import TrainerCallback
from pathlib import Path
import copy
from huggingface_hub import HfApi
from .utils import log_job_metrics

class SQLiteLoggingCallback(TrainerCallback):
    def __init__(self, job_id: str):
        self.job_id = job_id

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            loss = logs.get("loss") or logs.get("train_loss") or 0.0
            reward = logs.get("reward") or logs.get("rewards/chosen") or logs.get("rewards/mean") or logs.get("mean_reward")
            step = state.global_step
            epoch = state.epoch or 0.0
            
            # Extract any other scalar metrics
            extra = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            
            log_job_metrics(
                job_id=self.job_id,
                step=step,
                epoch=epoch,
                loss=loss,
                reward=reward,
                extra_metrics=extra
            )

# --- Verifiable Reward Functions for RLVR and GRPO ---

def format_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Verify that the model outputs structured reasoning in XML format."""
    rewards = []
    for completion in completions:
        has_reasoning = "<reasoning>" in completion and "</reasoning>" in completion
        has_answer = "<answer>" in completion and "</answer>" in completion
        rewards.append(1.0 if (has_reasoning and has_answer) else 0.0)
    return rewards

def accuracy_reward_func(prompts, completions, ground_truth, **kwargs) -> list[float]:
    """Verify that the output contains the correct ground-truth target."""
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        clean_completion = completion.strip().lower()
        if gt.strip().lower() in clean_completion:
            rewards.append(1.5)
        else:
            rewards.append(0.0)
    return rewards


def _run_text_finetuning_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    method = config.train.peft.method
    if method == "qlora" and config.trainer.device != "cuda":
        warnings.warn(
            "QLoRA is only available on CUDA devices. Falling back to LoRA."
        )
        config.train.peft.method = "lora"
        method = "lora"

    logging.info(
        f"--- Starting Text-Only Preference/Fine-tuning with {method.upper()} ---"
    )
    output_dir = config.get_output_dir()
    dataset = prepare_text_dataset(config)

    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if method == "qlora"
        else None
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model.repo_id,
        quantization_config=bnb_config,
        device_map=config.trainer.device,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if config.trainer.device == "cuda" else "eager",
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.repo_id, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=config.train.peft.r,
        lora_alpha=config.train.peft.lora_alpha,
        lora_dropout=config.train.peft.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=(method == "dora"),
    )

    # Configure Logging / MLflow environment
    if config.trainer.logger == "mlflow":
        os.environ["MLFLOW_TRACKING_URI"] = config.trainer.mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
        os.environ["MLFLOW_EXPERIMENT_NAME"] = output_dir.name
        os.environ["MLFLOW_RUN_NAME"] = f"run_{output_dir.name}"

    sqlite_callback = SQLiteLoggingCallback(output_dir.name)

    if method == "dpo":
        logging.info("Initializing TRL DPOTrainer...")
        training_args = DPOConfig(
            output_dir=str(output_dir),
            num_train_epochs=config.trainer.max_epochs,
            per_device_train_batch_size=config.train.batch_size,
            gradient_accumulation_steps=1,
            optim="paged_adamw_32bit" if config.trainer.device == "cuda" else "adamw_torch",
            learning_rate=config.train.llm_lr,
            bf16=(config.trainer.device == "cuda"),
            fp16=False,
            logging_steps=5,
            do_eval=config.trainer.evaluation.do_eval,
            eval_strategy="epoch" if config.trainer.evaluation.do_eval and dataset.get("test") else "no",
            save_strategy="epoch",
            report_to="mlflow" if config.trainer.logger == "mlflow" else "none",
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            peft_config=peft_config,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("test"),
            processing_class=tokenizer,
            callbacks=[sqlite_callback],
        )

    elif method in ["grpo", "rlvr"]:
        logging.info("Initializing TRL GRPOTrainer for Verifiable Reward Policy Optimization...")
        training_args = GRPOConfig(
            output_dir=str(output_dir),
            num_train_epochs=config.trainer.max_epochs,
            per_device_train_batch_size=config.train.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=config.train.llm_lr,
            bf16=(config.trainer.device == "cuda"),
            fp16=False,
            logging_steps=5,
            report_to="mlflow" if config.trainer.logger == "mlflow" else "none",
        )
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[format_reward_func, accuracy_reward_func],
            peft_config=peft_config,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("test"),
            processing_class=tokenizer,
            callbacks=[sqlite_callback],
        )

    else:
        # Standard SFT
        logging.info("Initializing TRL SFTTrainer...")
        training_args = SFTConfig(
            output_dir=str(output_dir),
            dataset_text_field="completion",
            num_train_epochs=config.trainer.max_epochs,
            per_device_train_batch_size=config.train.batch_size,
            gradient_accumulation_steps=1,
            optim="paged_adamw_32bit" if config.trainer.device == "cuda" else "adamw_torch",
            learning_rate=config.train.llm_lr,
            bf16=(config.trainer.device == "cuda"),
            fp16=False,
            logging_steps=5,
            do_eval=config.trainer.evaluation.do_eval,
            eval_strategy="epoch" if config.trainer.evaluation.do_eval and dataset.get("test") else "no",
            save_strategy="epoch",
            load_best_model_at_end=bool(config.trainer.evaluation.do_eval and dataset.get("test")),
            report_to="mlflow" if config.trainer.logger == "mlflow" else "none",
            push_to_hub=config.train.push_to_hub,
            hub_model_id=config.train.hub_model_id,
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("test"),
            peft_config=peft_config,
            args=training_args,
            callbacks=[sqlite_callback],
        )

    logging.info("Starting trainer training...")
    trainer.train()
    
    if config.trainer.evaluation.do_eval and dataset.get("test") and method not in ["grpo", "rlvr"]:
        logging.info("Evaluating final model...")
        metrics = trainer.evaluate()
        if 'eval_loss' in metrics:
            logging.info(f"Evaluation results: Perplexity: {math.exp(metrics['eval_loss']):.2f}")

    final_adapter_path = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter_path))

    if config.train.push_to_hub:
         trainer.push_to_hub()
         logging.info(f"Pushed to hub: {config.train.hub_model_id}")

    logging.info(f"--- Preference/Finetuning Complete. Adapter saved to: {final_adapter_path} ---")
    return {"path": final_adapter_path, "metrics": metrics if 'metrics' in locals() else {}}


def _run_multimodal_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    logging.info("--- Starting Multi-Modal Finetuning ---")
    output_dir = config.get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.repo_id,
        cache_dir=config.model.base_model_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collate_fn = partial(multimodal_collate_fn, tokenizer=tokenizer)

    if config.data.dataset_repo_id:
        # Streaming path
        train_dataset = StreamingMultiModalDataset(config, tokenizer)
        val_dataset = None
        if config.trainer.evaluation.do_eval:
             try:
                 available_splits = get_dataset_splits(config.data.dataset_repo_id)
                 eval_split = None
                 for split in ["test", "validation"]:
                     if split in available_splits:
                         eval_split = split
                         break

                 if eval_split:
                     val_config = copy.deepcopy(config)
                     val_config.data.split = eval_split
                     val_dataset = StreamingMultiModalDataset(val_config, tokenizer)
                 else:
                     warnings.warn("Could not find 'test' or 'validation' split. Evaluation disabled.")

             except Exception as e:
                 warnings.warn(f"Error checking splits: {e}. Evaluation disabled.")

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=config.train.batch_size,
                collate_fn=collate_fn,
                num_workers=4,
                pin_memory=True,
                shuffle=False,
            )
            if val_dataset
            else None
        )
    else:
        # Local file path
        full_dataset = MultiModalDataset(config, tokenizer)
        train_size = len(full_dataset)
        val_size = (
            int(train_size * config.trainer.evaluation.eval_dataset_size)
            if config.trainer.evaluation.do_eval
            else 0
        )

        train_dataset, val_dataset = (
            (torch.utils.data.random_split(full_dataset, [train_size - val_size, val_size]))
            if val_size > 0 and train_size > val_size
            else (full_dataset, None)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=config.train.batch_size,
                collate_fn=collate_fn,
                num_workers=4,
                pin_memory=True,
            )
            if val_dataset
            else None
        )

    # Check for Native Multimodal Support
    hf_config = AutoConfig.from_pretrained(config.model.repo_id, trust_remote_code=True)
    is_native_vlm = hasattr(hf_config, "vision_config") and hf_config.vision_config is not None

    model = MultimodalLLM(config)

    if config.trainer.logger == "mlflow":
        from lightning.pytorch.loggers import MLFlowLogger
        tracking_uri = config.trainer.mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
        logger = MLFlowLogger(experiment_name=output_dir.name, tracking_uri=tracking_uri)
    elif config.trainer.logger == "tensorboard":
        logger = TensorBoardLogger("logs", name=output_dir.name)
    else:
        logger = CSVLogger("logs", name=output_dir.name)

    callbacks = []
    if config.trainer.checkpoint_callback:
        if val_loader:
             callbacks.append(
                ModelCheckpoint(
                    dirpath=output_dir,
                    monitor="val_loss",
                    mode="min",
                    save_top_k=1,
                    filename="best_model",
                )
             )
        else:
             pass

    precision = "bf16-mixed" if config.trainer.device == "cuda" else "32-true"

    trainer = L.Trainer(
        max_epochs=config.trainer.max_epochs,
        accelerator=config.trainer.device,
        devices=config.trainer.devices,
        precision=precision,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader, val_loader)

    
    # Save final model in HF format
    if config.train.push_to_hub and config.train.hub_model_id:
        pass # Push logic handles adapter
    
    # Always save local HF checkpoint
    model.save_multimodal_checkpoint(output_dir)
    best_path = output_dir

    if callbacks and hasattr(callbacks[0], "best_model_path") and callbacks[0].best_model_path:
         # Does ModelCheckpoint save HF format? No, usually .ckpt
         # We might want to load best .ckpt and then save as HF?
         # For simplicity, we save final state as HF. 
         # If user wants best, they can load .ckpt and convert.
         pass

    preprocessors = {
        "scaler": getattr(train_dataset, "scaler", None) or getattr(full_dataset if 'full_dataset' in locals() else None, "scaler", None),
        "cat_mappings": getattr(train_dataset, "cat_mappings", {}) or getattr(full_dataset if 'full_dataset' in locals() else None, "cat_mappings", {}),
    }
    torch.save(preprocessors, best_path.parent / "preprocessors.pt")

    if config.train.push_to_hub and config.train.hub_model_id:
        logging.info(f"Pushing to hub: {config.train.hub_model_id}")
        try:
            # Need to create repo if not exists
            api = HfApi()
            api.create_repo(repo_id=config.train.hub_model_id, exist_ok=True)

            # Since we are using LightningModule, we need to extract the model and save as PEFT adapter if native
            # OR upload the checkpoint.
            # Usually users want the PEFT adapter.

            if model.is_native_vlm:
                # self.llm is a PeftModel
                model.llm.push_to_hub(config.train.hub_model_id)
            else:
                # Custom towers + frozen LLM.
                # We haven't implemented saving custom towers as HF model easily.
                # Usually we push the checkpoint.
                api.upload_folder(
                    folder_path=str(output_dir),
                    repo_id=config.train.hub_model_id,
                    repo_type="model",
                )
        except Exception as e:
            logging.error(f"Failed to push to hub: {e}")

    logging.info(f"--- Multi-Modal Finetuning Complete. Best model saved to: {best_path} ---")
    metrics = trainer.callback_metrics
    # Convert tensor metrics to float
    metrics = {k: v.item() if hasattr(v, "item") else v for k, v in metrics.items()}
    return {"path": best_path, "metrics": metrics}


def run_finetuning(config: PipelineConfig) -> Dict[str, Any]:
    torch.set_float32_matmul_precision("high")
    if config.is_multimodal:
        return _run_multimodal_pipeline(config)
    else:
        return _run_text_finetuning_pipeline(config)
