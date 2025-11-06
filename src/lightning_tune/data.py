import torch, polars as pl
from pathlib import Path
from typing import Dict
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from .config import PipelineConfig
from datasets import load_dataset, DatasetDict


def prepare_text_dataset(config: PipelineConfig) -> DatasetDict:
    def format_prompt(example):
        if config.data.input_column and example.get(config.data.input_column):
            return {
                "text": f"### Instruction:\n{example.get(config.data.instruction_column, '')}\n\n### Input:\n{example.get(config.data.input_column, '')}\n\n### Response:\n{example.get(config.data.output_column, '')}"
            }
        return {
            "text": f"### Instruction:\n{example.get(config.data.instruction_column, '')}\n\n### Response:\n{example.get(config.data.output_column, '')}"
        }

    file_type = config.data.file_path.suffix.lower().replace(".", "")
    dataset = load_dataset(
        file_type, data_files=str(config.data.file_path), split="train"
    ).map(format_prompt)
    return (
        dataset.train_test_split(test_size=config.trainer.evaluation.eval_dataset_size)
        if config.trainer.evaluation.do_eval
        and config.trainer.evaluation.eval_dataset_size > 0
        else DatasetDict({"train": dataset})
    )


class MultiModalDataset(Dataset):
    def __init__(self, config: PipelineConfig, tokenizer: AutoTokenizer):
        self.config, self.tokenizer = config.data, tokenizer
        self.df = pl.read_csv(self.config.file_path)
        self._setup_preprocessors()
        self._setup_image_transforms()

    def _setup_image_transforms(self):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _setup_preprocessors(self):
        self.scaler, self.cat_mappings = None, {}
        if self.config.tabular_config:
            tc = self.config.tabular_config
            if tc.numerical_columns:
                self.scaler = StandardScaler().fit(
                    self.df.select(tc.numerical_columns).to_numpy()
                )
            if tc.categorical_columns:
                self.cat_mappings = {
                    c: {v: i for i, v in enumerate(self.df[c].unique())}
                    for c in tc.categorical_columns
                }
                tc.categorical_cardinality = {
                    c: len(m) for c, m in self.cat_mappings.items()
                }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int) -> Dict:
        row = self.df.row(i, named=True)
        text, target = " ".join(str(row[c]) for c in self.config.text_columns), str(
            row[self.config.output_column]
        )
        source, target_toks = self.tokenizer.encode(text), self.tokenizer.encode(target)
        input_ids, labels = source + target_toks, torch.full(
            (len(source) + len(target_toks),), -100
        )
        labels[len(source) :] = torch.tensor(target_toks)
        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": labels,
        }
        if self.config.vision_config and self.config.image_root_path:
            img_p = (
                self.config.image_root_path
                / row[self.config.vision_config.image_column]
            )
            # <-- ADDED ROBUSTNESS CHECK -->
            try:
                item["image"] = self.image_transform(Image.open(img_p).convert("RGB"))
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Image not found at path: {img_p}. "
                    f"Please ensure your .zip file has an 'images/' folder and the paths in your CSV are correct."
                )
            # <-- END OF CHECK -->
            # item["image"] = self.image_transform(Image.open(img_p).convert("RGB"))
        if self.config.tabular_config:
            tc = self.config.tabular_config
            if tc.numerical_columns:
                item["tabular_num"] = torch.tensor(
                    self.scaler.transform([[row[c] for c in tc.numerical_columns]])[0],
                    dtype=torch.float32,
                )
            if tc.categorical_columns:
                item["tabular_cat"] = torch.tensor(
                    [self.cat_mappings[c][row[c]] for c in tc.categorical_columns],
                    dtype=torch.long,
                )
        return item


def multimodal_collate_fn(batch, tokenizer: AutoTokenizer):
    input_ids, labels = [item["input_ids"] for item in batch], [
        item["labels"] for item in batch
    ]
    max_len = max(len(ids) for ids in input_ids)
    collated = {
        "input_ids": torch.stack(
            [
                torch.nn.functional.pad(
                    ids, (0, max_len - len(ids)), value=tokenizer.pad_token_id
                )
                for ids in input_ids
            ]
        ),
        "labels": torch.stack(
            [
                torch.nn.functional.pad(l, (0, max_len - len(l)), value=-100)
                for l in labels
            ]
        ),
    }
    for key in ["image", "tabular_num", "tabular_cat"]:
        if key in batch[0]:
            collated[key] = torch.stack([item[key] for item in batch])
    return collated
