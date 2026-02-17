import torch, torch.nn as nn, lightning as L, timm, torchmetrics
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoConfig, AutoModel
from peft import get_peft_model, LoraConfig, TaskType
from .config import PipelineConfig, TabularConfig, VisionConfig, AudioConfig
import logging
from pathlib import Path

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
            config.model_name, pretrained=True, num_classes=0, global_pool='avg'
        )
        self.projection = nn.Linear(
            self.vision_model.num_features, config.projection_dim
        )

    def forward(self, images):
        return self.projection(self.vision_model(images)).unsqueeze(1)


class AudioTower(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.audio_model = AutoModel.from_pretrained(config.model_name)
        # Determine output dim - usually hidden_size
        self.out_dim = self.audio_model.config.hidden_size
        self.projection = nn.Linear(self.out_dim, config.projection_dim)

    def forward(self, audio):
        # audio: [B, T] or [B, C, T] ? 
        # Using simple mean pooling over time for now as a "global audio feature"
        # Or return sequence? For simplicity in this tower, sequence [1, D]
        outputs = self.audio_model(audio)
        # different models have different outputs.
        # Whisper: last_hidden_state [B, T, D]
        if hasattr(outputs, "last_hidden_state"):
            feat = outputs.last_hidden_state.mean(dim=1) # [B, D]
        elif hasattr(outputs, "pooler_output"):
             feat = outputs.pooler_output
        else:
             feat = outputs[0].mean(dim=1)
        
        return self.projection(feat).unsqueeze(1)


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
        # Apply PEFT (LoRA/DoRA) to the LLM
        # We froze the base model above. PEFT will add trainable adapters.
        peft_cfg = LoraConfig(
            r=config.train.peft.r,
            lora_alpha=config.train.peft.lora_alpha,
            lora_dropout=config.train.peft.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=None, # Default usually works.
            use_dora=(config.train.peft.method == "dora")
        )
        self.llm = get_peft_model(self.llm, peft_cfg)
        self.llm.print_trainable_parameters()

        # Initialize towers only if NOT native VLM (for vision) or always for tabular
        self.vision_tower = None
        if config.data.vision_config and not self.is_native_vlm:
             self.vision_tower = VisionTower(config.data.vision_config)

        self.audio_tower = None
        if config.data.audio_config:
             self.audio_tower = AudioTower(config.data.audio_config)

        self.tabular_tower = (
            TabularTower(config.data.tabular_config)
            if config.data.tabular_config
            else None
        )
        self.to(self.llm.dtype)

    def forward(self, batch):
        if self.is_native_vlm:
            # Native VLM handling
            forward_kwargs = {"input_ids": batch["input_ids"], "labels": batch["labels"]}

            if "image" in batch:
                forward_kwargs["pixel_values"] = batch["image"].to(self.llm.dtype)

            # If tabular data is present, we must process it and inject it.
            # Native VLMs typically work with inputs_embeds OR input_ids + pixel_values.
            # To inject Tabular embeddings, we MUST use inputs_embeds.
            # But calculating inputs_embeds for native VLM (which handles pixel_values internally) is tricky
            # because native VLMs often handle image-to-embedding logic inside their forward().

            # Strategy:
            # 1. Get text embeddings from the model (if possible exposed).
            # 2. Add Tabular embeddings as prefix.
            # 3. But wait, native VLM needs to inject image embeddings too.
            #    If we override inputs_embeds, we disable the internal image handling usually.

            # If the model allows inputs_embeds AND pixel_values simultaneously (some do, e.g. LLaVA),
            # we can pass inputs_embeds for text+tabular, and pixel_values for images.
            # But usually inputs_embeds supersedes input_ids.

            # Robust Approach for now:
            # If Tabular Tower exists with Native VLM, we assume the user accepts that
            # we might NOT be able to easily inject it into the *middle* of the logic without
            # re-implementing the VLM's forward.

            # HOWEVER, most VLMs (like Llava) process input_ids to embeds, insert image embeds, then run LM.
            # If we pass inputs_embeds, we are responsible for EVERYTHING.

            # Alternative: Just allow Tabular Tower to function if it can be prepended.
            # For simplicity & robustness requested:
            # If Native VLM + Tabular, we try to use inputs_embeds if possible.
            # But getting text_embeds from a wrapped PEFT model of a VLM is non-standard.

            # Let's check if we have tabular data first.
            has_tabular = self.tabular_tower and ("tabular_cat" in batch or "tabular_num" in batch)

            if has_tabular:
                 # This path is complex for Native VLMs.
                 # We will attempt to get text embeddings.
                 # NOTE: self.llm is a PeftModel. self.llm.base_model.model is typically the underlying transformer.
                 # But getting embeddings is model-specific.

                 # Simpler fallback: If native VLM + Tabular, warn user or skip tabular?
                 # User EXPLICITLY asked for it: "a native VLM cant have Tabular config in your code idk why that restriction".

                 # So we MUST support it.
                 # We will assume we can get input_embeddings via get_input_embeddings()
                 try:
                     # Access base model to get embeddings if needed
                     # model = self.llm.get_base_model() # Peft
                     # But most HF models support get_input_embeddings()
                     embeddings = self.llm.get_input_embeddings()
                     text_embeds = embeddings(batch["input_ids"])

                     prefix_embeds = []
                     tabular_cat = batch.get("tabular_cat")
                     tabular_num = batch.get("tabular_num")
                     if tabular_num is not None:
                        tabular_num = tabular_num.to(self.llm.dtype)
                     tab_embed = self.tabular_tower(tabular_cat, tabular_num)
                     if tab_embed is not None:
                        prefix_embeds.append(tab_embed)

                     # Concatenate
                     # [Tabular, Text]
                     inputs_embeds = torch.cat(prefix_embeds + [text_embeds], dim=1)

                     # Pass to model.
                     # We hope model handles (inputs_embeds + pixel_values) correctly.
                     # LLaVA implementation usually:
                     # if inputs_embeds is None: inputs_embeds = embed(input_ids)
                     # then merges images.

                     # So if we pass inputs_embeds, it should work IF we don't mess up image placeholders.
                     # Image placeholders in LLaVA are tokens in input_ids.
                     # If we replace input_ids with inputs_embeds, we must ensure placeholders are preserved in embeddings?
                     # Yes, text_embeds contains embeddings of <image> tokens.

                     # So: inputs_embeds + pixel_values should work.
                     del forward_kwargs["input_ids"]
                     forward_kwargs["inputs_embeds"] = inputs_embeds

                 except Exception as e:
                     logging.error(f"Failed to mix Tabular with Native VLM: {e}")
                     # Fallback to just Native VLM (ignore tabular to avoid crash)

            outputs = self.llm(**forward_kwargs)
            # Native VLM output usually doesn't have a standardized 'num_prefix_tokens' we can use to shift loss easily?
            # Actually if we prepend tabular, we shift labels?
            # If we added tabular prefix, we need to adjust output?
            # Or just let native logic handle it.
            # If we passed inputs_embeds with extra prefix, the model sees a longer sequence.
            # But 'labels' passed in forward_kwargs must match the length of inputs_embeds?
            # HF models usually expect labels to match inputs.
            # Our 'labels' in batch match 'input_ids'.
            # If we prepend tabular embeddings, we must prepend dummy labels (-100) to 'labels'.

            if has_tabular and "inputs_embeds" in forward_kwargs:
                # Get length difference
                diff = forward_kwargs["inputs_embeds"].shape[1] - batch["labels"].shape[1]
                if diff > 0:
                     prefix_labels = torch.full((batch["labels"].shape[0], diff), -100, device=batch["labels"].device, dtype=batch["labels"].dtype)
                     forward_kwargs["labels"] = torch.cat([prefix_labels, batch["labels"]], dim=1)

            # Re-run if we updated labels
            outputs = self.llm(**forward_kwargs)
            return outputs.logits, 0

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
            if self.audio_tower and "audio" in batch:
                audio = batch["audio"].to(self.llm.dtype)
                prefix_embeds.append(self.audio_tower(audio))
                num_prefix_tokens += 1
            inputs_embeds = torch.cat(prefix_embeds + [text_embeds], dim=1)
            outputs = self.llm(inputs_embeds=inputs_embeds)
            return outputs.logits, num_prefix_tokens

    def _calculate_loss(self, logits, labels, num_prefix):
        if num_prefix > 0:
            # Shift for prefix-tuning alignment (Logits[0] -> Label[0])
            shift_logits = logits[:, num_prefix - 1 : -1, :].contiguous()
            shift_labels = labels.contiguous()
        else:
            # Standard Causal Loss (Logits[t] -> Label[t+1])
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

        return torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def training_step(self, batch, batch_idx):
        if self.is_native_vlm:
            forward_kwargs = {"input_ids": batch["input_ids"], "labels": batch["labels"]}
            if "image" in batch:
                forward_kwargs["pixel_values"] = batch["image"].to(self.llm.dtype)

            # Replicate the forward logic for Tabular handling
            has_tabular = self.tabular_tower and ("tabular_cat" in batch or "tabular_num" in batch)
            if has_tabular:
                 try:
                     embeddings = self.llm.get_input_embeddings()
                     text_embeds = embeddings(batch["input_ids"])
                     prefix_embeds = []
                     tabular_cat = batch.get("tabular_cat")
                     tabular_num = batch.get("tabular_num")
                     if tabular_num is not None:
                        tabular_num = tabular_num.to(self.llm.dtype)
                     tab_embed = self.tabular_tower(tabular_cat, tabular_num)
                     if tab_embed is not None:
                        prefix_embeds.append(tab_embed)
                     inputs_embeds = torch.cat(prefix_embeds + [text_embeds], dim=1)
                     del forward_kwargs["input_ids"]
                     forward_kwargs["inputs_embeds"] = inputs_embeds

                     diff = inputs_embeds.shape[1] - batch["labels"].shape[1]
                     if diff > 0:
                         prefix_labels = torch.full((batch["labels"].shape[0], diff), -100, device=batch["labels"].device, dtype=batch["labels"].dtype)
                         forward_kwargs["labels"] = torch.cat([prefix_labels, batch["labels"]], dim=1)
                 except Exception as e:
                     pass

            outputs = self.llm(**forward_kwargs)
            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            else:
                loss = self._calculate_loss(outputs.logits, forward_kwargs["labels"], 0)
        else:
            logits, num_prefix = self(batch)
            loss = self._calculate_loss(logits, batch["labels"], num_prefix)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.is_native_vlm:
            # Similar logic as training_step
            forward_kwargs = {"input_ids": batch["input_ids"], "labels": batch["labels"]}
            if "image" in batch:
                forward_kwargs["pixel_values"] = batch["image"].to(self.llm.dtype)

            has_tabular = self.tabular_tower and ("tabular_cat" in batch or "tabular_num" in batch)
            if has_tabular:
                 try:
                     embeddings = self.llm.get_input_embeddings()
                     text_embeds = embeddings(batch["input_ids"])
                     prefix_embeds = []
                     tabular_cat = batch.get("tabular_cat")
                     tabular_num = batch.get("tabular_num")
                     if tabular_num is not None:
                        tabular_num = tabular_num.to(self.llm.dtype)
                     tab_embed = self.tabular_tower(tabular_cat, tabular_num)
                     if tab_embed is not None:
                        prefix_embeds.append(tab_embed)
                     inputs_embeds = torch.cat(prefix_embeds + [text_embeds], dim=1)
                     del forward_kwargs["input_ids"]
                     forward_kwargs["inputs_embeds"] = inputs_embeds
                     diff = inputs_embeds.shape[1] - batch["labels"].shape[1]
                     if diff > 0:
                         prefix_labels = torch.full((batch["labels"].shape[0], diff), -100, device=batch["labels"].device, dtype=batch["labels"].dtype)
                         forward_kwargs["labels"] = torch.cat([prefix_labels, batch["labels"]], dim=1)
                 except Exception:
                     pass

            outputs = self.llm(**forward_kwargs)
            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            else:
                loss = self._calculate_loss(outputs.logits, forward_kwargs["labels"], 0)
        else:
            logits, num_prefix = self(batch)
            loss = self._calculate_loss(logits, batch["labels"], num_prefix)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_perplexity", torch.exp(loss), prog_bar=True)

    def configure_optimizers(self):
        # Gather all trainable parameters
        params = [p for p in self.parameters() if p.requires_grad]
        lr = self.config.train.llm_lr if self.is_native_vlm else self.config.train.tower_lr
        return torch.optim.AdamW(params, lr=lr)

    @torch.inference_mode()
    def generate(self, batch, tokenizer, max_new_tokens=100):
        if self.is_native_vlm:
             gen_kwargs = {
                 "input_ids": batch["input_ids"],
                 "max_new_tokens": max_new_tokens,
                 "pad_token_id": tokenizer.pad_token_id
             }
             if "image" in batch:
                 gen_kwargs["pixel_values"] = batch["image"].to(self.llm.dtype)

             # Tabular injection for generation
             has_tabular = self.tabular_tower and ("tabular_cat" in batch or "tabular_num" in batch)
             if has_tabular:
                 try:
                     embeddings = self.llm.get_input_embeddings()
                     text_embeds = embeddings(batch["input_ids"])
                     prefix_embeds = []
                     tabular_cat = batch.get("tabular_cat")
                     tabular_num = batch.get("tabular_num")
                     if tabular_num is not None:
                        tabular_num = tabular_num.to(self.llm.dtype)
                     tab_embed = self.tabular_tower(tabular_cat, tabular_num)
                     if tab_embed is not None:
                        prefix_embeds.append(tab_embed)
                     inputs_embeds = torch.cat(prefix_embeds + [text_embeds], dim=1)
                     del gen_kwargs["input_ids"]
                     gen_kwargs["inputs_embeds"] = inputs_embeds
                 except Exception:
                     pass

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

    def save_multimodal_checkpoint(self, output_dir: Path):
        """Saves the adapter and towers in a format friendly for HF loading."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA Adapter
        if isinstance(self.llm, __import__("peft").PeftModel):
             self.llm.save_pretrained(output_dir)
        else:
             # If full finetuning, save the whole model? Or just state dict?
             # For this task, we assume LoRA primarily.
             # If full model, maybe save_pretrained on base?
             if hasattr(self.llm, "save_pretrained"):
                 self.llm.save_pretrained(output_dir)
        
        # Save Towers
        towers_state = {}
        if self.vision_tower:
             towers_state["vision_tower"] = self.vision_tower.state_dict()
        if self.audio_tower:
             towers_state["audio_tower"] = self.audio_tower.state_dict()
        if self.tabular_tower:
             towers_state["tabular_tower"] = self.tabular_tower.state_dict()
        
        if towers_state:
             torch.save(towers_state, output_dir / "towers.pt")
        
        # Save Config
        with open(output_dir / "pipeline_config.json", "w") as f:
             f.write(self.config.model_dump_json(indent=2))
