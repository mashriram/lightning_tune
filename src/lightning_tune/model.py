import torch, torch.nn as nn, lightning as L, timm, torchmetrics
from transformers import AutoModelForCausalLM
from .config import PipelineConfig, TabularConfig, VisionConfig


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
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.model.repo_id,
            cache_dir=config.model.base_model_dir,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        [p.requires_grad_(False) for p in self.llm.parameters()]
        self.vision_tower = (
            VisionTower(config.data.vision_config)
            if config.data.vision_config
            else None
        )
        self.tabular_tower = (
            TabularTower(config.data.tabular_config)
            if config.data.tabular_config
            else None
        )

    def forward(self, batch):
        text_embeds = self.llm.get_input_embeddings()(batch["input_ids"])
        prefix_embeds, num_prefix_tokens = [], 0
        if self.vision_tower and "image" in batch:
            prefix_embeds.append(self.vision_tower(batch["image"]))
            num_prefix_tokens += 1
        if self.tabular_tower and ("tabular_cat" in batch or "tabular_num" in batch):
            tab_embed = self.tabular_tower(
                batch.get("tabular_cat"), batch.get("tabular_num")
            )
            if tab_embed is not None:
                prefix_embeds.append(tab_embed)
                num_prefix_tokens += 1
        inputs_embeds = torch.cat(prefix_embeds + [text_embeds], dim=1)
        outputs = self.llm(inputs_embeds=inputs_embeds)
        return outputs.logits, num_prefix_tokens

    def _calculate_loss(self, logits, labels, num_prefix):
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
        logits, num_prefix = self(batch)
        loss = self._calculate_loss(logits, batch["labels"], num_prefix)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, num_prefix = self(batch)
        loss = self._calculate_loss(logits, batch["labels"], num_prefix)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_perplexity", torch.exp(loss), prog_bar=True)

    def configure_optimizers(self):
        params = list(
            self.vision_tower.parameters() if self.vision_tower else []
        ) + list(self.tabular_tower.parameters() if self.tabular_tower else [])
        return torch.optim.AdamW(params, lr=self.config.train.tower_lr)

    @torch.inference_mode()
    def generate(self, batch, tokenizer, max_new_tokens=100):
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
