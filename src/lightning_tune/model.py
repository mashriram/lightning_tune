import torch, torch.nn as nn, lightning as L, timm, torchmetrics
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
from .config import PipelineConfig, TabularConfig, VisionConfig
import logging

class TabularTower(nn.Module):
    def __init__(self, config: TabularConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.ModuleDict(
            {
                c: nn.Embedding(card, 32)
                for c, card in config.categorical_cardinality.items()
            }
        )
        input_size = (len(config.categorical_columns) * 32) + len(
            config.numerical_columns
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_size, config.projection_dim // 2),
            nn.ReLU(),
            nn.Linear(config.projection_dim // 2, config.projection_dim),
        )

    def forward(self, x_cat, x_num):
        cat_embeds = (
            [
                self.embeddings[c](x_cat[:, i])
                for i, c in enumerate(self.config.categorical_columns)
            ]
            if self.config.categorical_columns and x_cat is not None
            else []
        )
        tensors = cat_embeds + (
            [x_num] if self.config.numerical_columns and x_num is not None else []
        )
        if not tensors:
            return None
        combined = torch.cat(tensors, dim=1) if len(tensors) > 1 else tensors[0]
        return self.mlp(combined).unsqueeze(1)


class VisionTower(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.vision_model = timm.create_model(
            config.model_name, pretrained=True, num_classes=0
        )
        self.projection = nn.Linear(
            self.vision_model.num_features, config.projection_dim
        )

    def forward(self, images):
        return self.projection(self.vision_model(images)).unsqueeze(1)


class MultimodalLLM(L.LightningModule):
    def __init__(self, config: PipelineConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.model_dump())

        # Determine if we should use native Vision support
        hf_config = AutoConfig.from_pretrained(config.model.repo_id, trust_remote_code=True)
        self.is_native_vlm = False

        # Check for vision config or architectures suggesting VLM
        if (hasattr(hf_config, "vision_config") and hf_config.vision_config) or \
           (hasattr(hf_config, "architectures") and any("Llava" in a or "Idefics" in a for a in hf_config.architectures or [])):
            self.is_native_vlm = True
            logging.info(f"Model {config.model.repo_id} detected as native VLM. Using AutoModelForVision2Seq/CausalLM without extra vision tower.")

            # Use Vision2Seq or fallback to CausalLM (some VLMs are loaded via CausalLM)
            # Safest bet is usually AutoModelForCausalLM if trust_remote_code=True for many recent VLMs (like Llava-Next)
            # but some require AutoModelForVision2Seq.
            # We try generic auto class first or fallback to causal LM.
            try:
                self.llm = AutoModelForVision2Seq.from_pretrained(
                    config.model.repo_id,
                    cache_dir=config.model.base_model_dir,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
                )
            except Exception:
                self.llm = AutoModelForCausalLM.from_pretrained(
                    config.model.repo_id,
                    cache_dir=config.model.base_model_dir,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
                )
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                config.model.repo_id,
                cache_dir=config.model.base_model_dir,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
            )

        # Freeze LLM weights usually
        [p.requires_grad_(False) for p in self.llm.parameters()]

        # If native VLM, we should apply PEFT (LoRA) immediately since we froze everything.
        # Otherwise we have no trainable params.
        if self.is_native_vlm:
             # Map our PeftConfig to Peft LoraConfig
             peft_cfg = LoraConfig(
                 r=config.train.peft.r,
                 lora_alpha=config.train.peft.lora_alpha,
                 lora_dropout=config.train.peft.lora_dropout,
                 bias="none",
                 task_type=TaskType.CAUSAL_LM, # Or specific type? Causal LM usually safe for next token prediction VLMs
                 target_modules=None # Default usually works, or 'q_proj', 'v_proj'
             )
             self.llm = get_peft_model(self.llm, peft_cfg)
             self.llm.print_trainable_parameters()

        # Initialize towers only if NOT native VLM (for vision) or always for tabular
        self.vision_tower = None
        if config.data.vision_config and not self.is_native_vlm:
             self.vision_tower = VisionTower(config.data.vision_config)

        self.tabular_tower = (
            TabularTower(config.data.tabular_config)
            if config.data.tabular_config
            else None
        )
        self.to(self.llm.dtype)

    def forward(self, batch):
        if self.is_native_vlm:
            # Native VLM handling
            # Typically expects 'pixel_values' or 'images' in inputs
            # Our dataset provides 'image' tensor.
            # We need to map batch keys to what model expects.
            forward_kwargs = {"input_ids": batch["input_ids"], "labels": batch["labels"]}

            if "image" in batch:
                # Some models expect 'pixel_values', others 'images'
                # Llava usually expects 'images' (raw tensors) or 'pixel_values' if processed.
                # Since we used a simple resize/normalize transform, we likely match standard expected input or need processor.
                # Ideally, native VLMs should use their own Processor in dataset.
                # BUT, we are reusing our dataset logic.
                # Assuming 'pixel_values' is the standard key for HF Vision models.
                forward_kwargs["pixel_values"] = batch["image"].to(self.llm.dtype)

            outputs = self.llm(**forward_kwargs)
            return outputs.logits, 0 # No extra prefix tokens manually added

        else:
            # Custom Multimodal Logic
            text_embeds = self.llm.get_input_embeddings()(batch["input_ids"])
            prefix_embeds, num_prefix_tokens = [], 0
            if self.vision_tower and "image" in batch:
                prefix_embeds.append(self.vision_tower(batch["image"].to(self.llm.dtype)))
                num_prefix_tokens += 1
            if self.tabular_tower and ("tabular_cat" in batch or "tabular_num" in batch):
                tabular_cat = batch.get("tabular_cat")
                tabular_num = batch.get("tabular_num")
                if tabular_num is not None:
                    tabular_num = tabular_num.to(self.llm.dtype)
                tab_embed = self.tabular_tower(tabular_cat, tabular_num)
                if tab_embed is not None:
                    prefix_embeds.append(tab_embed)
                    num_prefix_tokens += 1
            inputs_embeds = torch.cat(prefix_embeds + [text_embeds], dim=1)
            outputs = self.llm(inputs_embeds=inputs_embeds)
            return outputs.logits, num_prefix_tokens

    def _calculate_loss(self, logits, labels, num_prefix):
        # Native VLMs usually compute loss internally if labels provided
        # But if we want to be uniform:
        if self.is_native_vlm:
             # If native VLM computed loss (it usually does if labels passed), return it?
             # Wait, forward() returned logits.
             pass

        shift_logits, shift_labels = (
            logits[:, num_prefix - 1 : -1, :].contiguous(),
            labels.contiguous(),
        )
        return torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def training_step(self, batch, batch_idx):
        if self.is_native_vlm:
            # Pass labels to forward to get loss directly if possible?
            # Or use manual calc.
            # Let's try to see if model outputs loss
            forward_kwargs = {"input_ids": batch["input_ids"], "labels": batch["labels"]}
            if "image" in batch:
                forward_kwargs["pixel_values"] = batch["image"].to(self.llm.dtype)

            outputs = self.llm(**forward_kwargs)
            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            else:
                loss = self._calculate_loss(outputs.logits, batch["labels"], 0)
        else:
            logits, num_prefix = self(batch)
            loss = self._calculate_loss(logits, batch["labels"], num_prefix)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.is_native_vlm:
            forward_kwargs = {"input_ids": batch["input_ids"], "labels": batch["labels"]}
            if "image" in batch:
                forward_kwargs["pixel_values"] = batch["image"].to(self.llm.dtype)
            outputs = self.llm(**forward_kwargs)
            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            else:
                loss = self._calculate_loss(outputs.logits, batch["labels"], 0)
        else:
            logits, num_prefix = self(batch)
            loss = self._calculate_loss(logits, batch["labels"], num_prefix)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_perplexity", torch.exp(loss), prog_bar=True)

    def configure_optimizers(self):
        # Gather all trainable parameters
        # This covers:
        # 1. Custom vision/tabular towers (if they exist)
        # 2. Native VLM LoRA adapters (since we applied PEFT, their params require_grad=True)
        # 3. Any other unfrozen params

        params = [p for p in self.parameters() if p.requires_grad]

        # Use LLM LR for LoRA params and Tower LR for towers?
        # For simplicity in this robust implementation, we use a single LR or default to LLM LR if native VLM.
        # Ideally we'd use groups.
        lr = self.config.train.llm_lr if self.is_native_vlm else self.config.train.tower_lr

        return torch.optim.AdamW(params, lr=lr)

    @torch.inference_mode()
    def generate(self, batch, tokenizer, max_new_tokens=100):
        if self.is_native_vlm:
             # Native generation
             gen_kwargs = {
                 "input_ids": batch["input_ids"],
                 "max_new_tokens": max_new_tokens,
                 "pad_token_id": tokenizer.pad_token_id
             }
             if "image" in batch:
                 gen_kwargs["pixel_values"] = batch["image"].to(self.llm.dtype)

             return self.llm.generate(**gen_kwargs)
        else:
            text_embeds = self.llm.get_input_embeddings()(batch["input_ids"])
            prefix_embeds = []
            if self.vision_tower and "image" in batch:
                prefix_embeds.append(self.vision_tower(batch["image"]))
            if self.tabular_tower and ("tabular_cat" in batch or "tabular_num" in batch):
                tab_embed = self.tabular_tower(
                    batch.get("tabular_cat"), batch.get("tabular_num")
                )
                if tab_embed is not None:
                    prefix_embeds.append(tab_embed)

            inputs_embeds = torch.cat(prefix_embeds + [text_embeds], dim=1)

            # Use the transformer's generate method which is highly optimized
            output_tokens = self.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Slice off the input tokens to return only the generated part
            return output_tokens[:, inputs_embeds.shape[1] :]
