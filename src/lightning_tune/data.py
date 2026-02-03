import torch, polars as pl, logging, warnings
from pathlib import Path
from typing import Dict, Union, Any, Iterator
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoTokenizer
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from .config import PipelineConfig
from datasets import load_dataset, DatasetDict
import io
import requests

def prepare_text_dataset(config: PipelineConfig) -> DatasetDict:
    def format_prompt(example):
        if config.data.input_column and example.get(config.data.input_column):
            return {
                "text": f"### Instruction:\n{example.get(config.data.instruction_column, '')}\n\n### Input:\n{example.get(config.data.input_column, '')}\n\n### Response:\n{example.get(config.data.output_column, '')}"
            }
        return {
            "text": f"### Instruction:\n{example.get(config.data.instruction_column, '')}\n\n### Response:\n{example.get(config.data.output_column, '')}"
        }

    if config.data.dataset_repo_id:
        # Streaming from Hugging Face
        logging.info(f"Streaming dataset from HF: {config.data.dataset_repo_id}")
        dataset = load_dataset(config.data.dataset_repo_id, split=config.data.split, streaming=True)
        dataset = dataset.map(format_prompt)

        if config.trainer.evaluation.do_eval:
            # Try to find a validation or test split
            eval_split = None
            try:
                # We can't easily check available splits in streaming mode without inspection,
                # but we can try-catch loading 'test' or 'validation'
                # For simplicity, we'll assume 'test' or 'validation' exists if user wants eval,
                # or we just rely on the user to have provided a split name if they wanted specific split.
                # But here we are looking for a separate eval split.

                # Let's try 'test', then 'validation'
                for split_name in ["test", "validation"]:
                    try:
                        eval_dataset = load_dataset(config.data.dataset_repo_id, split=split_name, streaming=True)
                        # If successful, use it
                        eval_dataset = eval_dataset.map(format_prompt)
                        logging.info(f"Using '{split_name}' split for evaluation.")
                        return DatasetDict({"train": dataset, "test": eval_dataset})
                    except:
                        continue

                # If we are here, we didn't find a split.
                # We can simulate a split by taking/skipping if needed, but for now just warn.
                warnings.warn("Could not find 'test' or 'validation' split for streaming dataset. Evaluation disabled.")
                return DatasetDict({"train": dataset})

            except Exception as e:
                warnings.warn(f"Error setting up evaluation for streaming dataset: {e}")
                return DatasetDict({"train": dataset})

        return DatasetDict({"train": dataset})

    else:
        # Local File
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
            try:
                item["image"] = self.image_transform(Image.open(img_p).convert("RGB"))
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Image not found at path: {img_p}. "
                    f"Please ensure your .zip file has an 'images/' folder and the paths in your CSV are correct."
                )
            # <-- END OF CHECK -->
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


class StreamingMultiModalDataset(IterableDataset):
    def __init__(self, config: PipelineConfig, tokenizer: AutoTokenizer):
        self.config, self.tokenizer = config.data, tokenizer
        logging.info(f"Initializing Streaming Dataset from {self.config.dataset_repo_id}")
        self.dataset = load_dataset(self.config.dataset_repo_id, split=self.config.split, streaming=True)
        self._setup_image_transforms()

        if self.config.tabular_config and self.config.tabular_config.numerical_columns:
            warnings.warn("Numerical column scaling is not supported in streaming mode. Data will be used as-is.")

        # For categorical columns, we need mappings.
        # In streaming, we can't pre-calculate them unless we scan the dataset or they are provided.
        # We will assume simple hashing or just skip categorical encoding for now if not provided?
        # Or better, just support what we can.
        if self.config.tabular_config and self.config.tabular_config.categorical_columns:
             warnings.warn("Categorical column encoding is not supported in streaming mode unless mappings are pre-defined. (Not implemented)")

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

    def __iter__(self) -> Iterator[Dict]:
        for row in self.dataset:
            # Process text
            text = " ".join(str(row[c]) for c in self.config.text_columns if row.get(c) is not None)
            target = str(row.get(self.config.output_column, ""))

            source = self.tokenizer.encode(text)
            target_toks = self.tokenizer.encode(target)

            input_ids = source + target_toks
            labels = torch.full((len(input_ids),), -100)
            labels[len(source):] = torch.tensor(target_toks)

            item = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": labels,
            }

            # Process Image
            if self.config.vision_config:
                image_col = self.config.vision_config.image_column
                if image_col in row:
                    img_data = row[image_col]
                    img = None
                    if isinstance(img_data, Image.Image):
                        img = img_data
                    elif isinstance(img_data, str):
                        # Assume URL or Path
                        if img_data.startswith("http"):
                             try:
                                 response = requests.get(img_data, stream=True, timeout=5)
                                 img = Image.open(response.raw)
                             except Exception as e:
                                 logging.warning(f"Failed to fetch image {img_data}: {e}")
                                 continue # Skip this item? or error?
                        else:
                             # Local path? (Unlikely in pure streaming HF but possible)
                             pass

                    if img:
                        item["image"] = self.image_transform(img.convert("RGB"))

            # Process Tabular (Numerical only for now, unscaled)
            if self.config.tabular_config and self.config.tabular_config.numerical_columns:
                 # We can't scale, but we can return the raw values
                 nums = [float(row[c]) for c in self.config.tabular_config.numerical_columns if row.get(c) is not None]
                 item["tabular_num"] = torch.tensor(nums, dtype=torch.float32)

            yield item


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
