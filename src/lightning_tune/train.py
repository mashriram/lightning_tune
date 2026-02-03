import lightning as L, torch, math, warnings, logging
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
from peft import LoraConfig
from trl import SFTTrainer
from pathlib import Path
import copy


def _run_text_finetuning_pipeline(config: PipelineConfig) -> Path:
    if config.train.peft.method == "qlora" and config.trainer.device != "cuda":
        warnings.warn(
            "QLoRA is only available on CUDA devices. Falling back to LoRA."
        )
        config.train.peft.method = "lora"

    logging.info(
        f"--- Starting Text-Only Finetuning with {config.train.peft.method.upper()} ---"
    )
    output_dir = config.get_output_dir()
    dataset = prepare_text_dataset(config)

    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if config.train.peft.method == "qlora"
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
        use_dora=(config.train.peft.method == "dora"),
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.trainer.max_epochs,
        per_device_train_batch_size=config.train.batch_size,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit" if config.trainer.device == "cuda" else "adamw_torch",
        learning_rate=config.train.llm_lr,
        bf16=(config.trainer.device == "cuda"),
        fp16=False,
        logging_steps=10,
        do_eval=config.trainer.evaluation.do_eval,
        eval_strategy=(
            "epoch"
            if config.trainer.evaluation.do_eval and dataset.get("test")
            else "no"
        ),
        save_strategy="epoch",
        load_best_model_at_end=config.trainer.evaluation.do_eval
        and dataset.get("test"),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test"),
        peft_config=peft_config,
        args=training_args,
    )

    logging.info("Starting SFTTrainer training...")
    trainer.train()
    if config.trainer.evaluation.do_eval and dataset.get("test"):
        logging.info("Evaluating final model...")
        metrics = trainer.evaluate()
        logging.info(f"Evaluation results: Perplexity: {math.exp(metrics['eval_loss']):.2f}")

    final_adapter_path = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter_path))
    logging.info(f"--- Finetuning Complete. Adapter saved to: {final_adapter_path} ---")
    return final_adapter_path


def _run_multimodal_pipeline(config: PipelineConfig) -> Path:
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

    logger = (
        TensorBoardLogger("logs", name=output_dir.name)
        if config.trainer.logger == "tensorboard"
        else CSVLogger("logs", name=output_dir.name)
    )

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

    if callbacks and hasattr(callbacks[0], "best_model_path") and callbacks[0].best_model_path:
        best_path = Path(callbacks[0].best_model_path)
    else:
        best_path = output_dir / "final.ckpt"
        trainer.save_checkpoint(str(best_path))

    preprocessors = {
        "scaler": getattr(train_dataset, "scaler", None) or getattr(full_dataset if 'full_dataset' in locals() else None, "scaler", None),
        "cat_mappings": getattr(train_dataset, "cat_mappings", {}) or getattr(full_dataset if 'full_dataset' in locals() else None, "cat_mappings", {}),
    }
    torch.save(preprocessors, best_path.parent / "preprocessors.pt")
    logging.info(f"--- Multi-Modal Finetuning Complete. Best model saved to: {best_path} ---")
    return best_path


def run_finetuning(config: PipelineConfig) -> Path:
    torch.set_float32_matmul_precision("high")
    if config.is_multimodal:
        return _run_multimodal_pipeline(config)
    else:
        return _run_text_finetuning_pipeline(config)
