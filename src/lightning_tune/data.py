import torch, polars as pl, logging, warnings
from pathlib import Path
from typing import Dict, Union, Any, Iterator
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoTokenizer
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from .config import PipelineConfig
from .hf_utils import get_dataset_splits
from datasets import load_dataset, DatasetDict
import io
import requests
import torchaudio
from torchaudio.transforms import Resample

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
            # Check available splits
            try:
                available_splits = get_dataset_splits(config.data.dataset_repo_id)
                eval_split = None
                for split in ["test", "validation"]:
                    if split in available_splits:
                        eval_split = split
                        break

                if eval_split:
                    logging.info(f"Using '{eval_split}' split for evaluation.")
                    eval_dataset = load_dataset(config.data.dataset_repo_id, split=eval_split, streaming=True)
                    eval_dataset = eval_dataset.map(format_prompt)
                    return DatasetDict({"train": dataset, "test": eval_dataset})
                else:
                    warnings.warn("Could not find 'test' or 'validation' split. Evaluation disabled.")
                    return DatasetDict({"train": dataset})

            except Exception as e:
                warnings.warn(f"Error checking splits: {e}. Evaluation disabled.")
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
        self.df = self._load_and_concat_datasets()
        self._setup_preprocessors()
        self._setup_image_transforms()
        self._setup_audio_transforms()

    def _load_and_concat_datasets(self) -> pl.DataFrame:
        dfs = []
        # Support legacy single-file if 'datasets' is empty (though config should have consolidated it)
        if not self.config.datasets and self.config.file_path:
             dfs.append(pl.read_csv(self.config.file_path))
        
        for ds_cfg in self.config.datasets:
            if ds_cfg.file_path:
                try:
                    df = pl.read_csv(ds_cfg.file_path)
                    dfs.append(df)
                except Exception as e:
                    logging.warning(f"Failed to load CSV {ds_cfg.file_path}: {e}")
            elif ds_cfg.repo_id:
                try:
                    # Load non-streaming for in-memory MultiModalDataset
                    from datasets import load_dataset
                    ds = load_dataset(
                        ds_cfg.repo_id, 
                        name=ds_cfg.config_name, 
                        split=ds_cfg.split or "train",
                        token=True # Uses env or cached
                    )
                    # Convert to Polars
                    # This might be heavy for large datasets, but MultiModalDataset implies in-memory
                    dfs.append(pl.from_arrow(ds.data.table))
                except Exception as e:
                     logging.warning(f"Failed to load HF dataset {ds_cfg.repo_id}: {e}")
        
        if not dfs:
             # Just raise warning and use empty DF if we want to allow empty init?
             # No, standard training requires data.
             if self.config.file_path: # Last ditch effort if consolidation failed or user bypassed
                 try:
                     return pl.read_csv(self.config.file_path)
                 except: pass
             raise ValueError("No data could be loaded from the provided configurations.")
        
        # Concat
        try:
            master_df = pl.concat(dfs, how="diagonal")
        except Exception as e:
            logging.warning(f"Could not confirm schemas for diagonal concat: {e}. Trying to coerce.")
            # If schemas differ significantly, this will fail.
            # We assume users provide relatively consistent data or we drop mismatch.
            master_df = dfs[0] 
        
        return master_df

    def _setup_audio_transforms(self):
        self.audio_transform = None
        if self.config.audio_config:
            # Simple resampling to 16kHz as default common base
            self.resample = Resample(orig_freq=48000, new_freq=16000) 
            pass

    def _load_audio(self, path: str):
        try:
            waveform, sample_rate = torchaudio.load(path, backend="soundfile")
            # Normalize/Resample if needed
            if sample_rate != 16000:
                resampler = Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
            return waveform.mean(0) # Mono
        except Exception as e:
            logging.warning(f"Error loading audio {path}: {e}")
            return torch.zeros(16000 * 10) # Dummy 10s silent

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
                # Check intersection with available columns
                available_nums = [c for c in tc.numerical_columns if c in self.df.columns]
                if available_nums:
                    self.scaler = StandardScaler().fit(
                        self.df.select(available_nums).to_numpy()
                    )
            if tc.categorical_columns:
                self.cat_mappings = {}
                for c in tc.categorical_columns:
                    if c in self.df.columns:
                         self.cat_mappings[c] = {v: i for i, v in enumerate(self.df[c].unique())}
                tc.categorical_cardinality = {
                    c: len(m) for c, m in self.cat_mappings.items()
                }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int) -> Dict:
        row = self.df.row(i, named=True)
        
        # Handle Alpaca/Instruction format robustly
        if self.config.instruction_column and row.get(self.config.instruction_column):
            instr = row.get(self.config.instruction_column, "")
            inp = row.get(self.config.input_column, "") if self.config.input_column else ""
            if inp:
                text = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n"
            else:
                text = f"### Instruction:\n{instr}\n\n### Response:\n"
        else:
             # Fallback to text columns
             valid_text_cols = [c for c in self.config.text_columns if c in row and row[c] is not None]
             text = " ".join(str(row[c]) for c in valid_text_cols)

        target = str(row.get(self.config.output_column, ""))
        source, target_toks = self.tokenizer.encode(text), self.tokenizer.encode(target)
        input_ids = source + target_toks
        labels = torch.full((len(input_ids),), -100)
        labels[len(source) :] = torch.tensor(target_toks)
        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": labels,
        }
        if self.config.audio_config:
             audio_col = self.config.audio_config.audio_column
             if audio_col in row and row[audio_col]:
                 audio_path = row[audio_col]
                 if self.config.image_root_path: 
                     if not Path(audio_path).exists() and self.config.image_root_path:
                         audio_path = str(self.config.image_root_path / audio_path)
                 
                 item["audio"] = self._load_audio(str(audio_path))
             else:
                 item["audio"] = torch.zeros(16000 * 10) 

        if self.config.vision_config:
            img_col = self.config.vision_config.image_column
            if img_col in row and row[img_col]:
                img_p = row[img_col]
                if self.config.image_root_path:
                     img_p = self.config.image_root_path / img_p
                
                try:
                    item["image"] = self.image_transform(Image.open(img_p).convert("RGB"))
                except (FileNotFoundError, OSError):
                    # Robustness: log warning but don't crash? or crash?
                    # For E2E robustness, maybe return black image and log?
                    logging.warning(f"Image not found/valid: {img_p}")
                    item["image"] = torch.zeros((3, 224, 224))

        if self.config.tabular_config:
            tc = self.config.tabular_config
            if tc.numerical_columns and self.scaler:
                valid_nums = [float(row.get(c, 0.0)) for c in tc.numerical_columns]
                item["tabular_num"] = torch.tensor(
                    self.scaler.transform([valid_nums])[0],
                    dtype=torch.float32,
                )
            if tc.categorical_columns:
                cats = []
                for c in tc.categorical_columns:
                    val = row.get(c)
                    mapping = self.cat_mappings.get(c, {})
                    cats.append(mapping.get(val, 0)) # 0 as unk
                item["tabular_cat"] = torch.tensor(cats, dtype=torch.long)
        return item


class StreamingMultiModalDataset(IterableDataset):
    def __init__(self, config: PipelineConfig, tokenizer: AutoTokenizer):
        self.config, self.tokenizer = config.data, tokenizer
        self._setup_image_transforms()
        
        from datasets import load_dataset, concatenate_datasets
        
        streams = []
        for ds_cfg in self.config.datasets:
            if ds_cfg.repo_id:
                try:
                     logging.info(f"Streaming dataset from HF: {ds_cfg.repo_id}")
                     ds = load_dataset(ds_cfg.repo_id, name=ds_cfg.config_name, split=ds_cfg.split or "train", streaming=True)
                     streams.append(ds)
                except Exception as e:
                     logging.warning(f"Failed to stream dataset {ds_cfg.repo_id}: {e}")
        
        # Legacy fallback if datasets list is empty (though config consolidation should handle this)
        if not streams and self.config.dataset_repo_id:
             try:
                 logging.info(f"Streaming dataset from HF (Legacy): {self.config.dataset_repo_id}")
                 streams.append(load_dataset(self.config.dataset_repo_id, name=self.config.dataset_config_name, split=self.config.split, streaming=True))
             except Exception as e:
                 logging.warning(f"Failed to stream legacy dataset {self.config.dataset_repo_id}: {e}")

        if not streams:
             raise ValueError("No streaming datasets available.")

        self.dataset = concatenate_datasets(streams)

        if self.config.tabular_config and self.config.tabular_config.numerical_columns:
            warnings.warn("Numerical column scaling is not supported in streaming mode. Data will be used as-is.")

        # For categorical columns, we need mappings.
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

            # Process Audio
            if self.config.audio_config:
                audio_col = self.config.audio_config.audio_column
                if audio_col in row:
                    audio_data = row[audio_col]
                    # If it's bytes/dict from HF dataset (Audio feature)
                    if isinstance(audio_data, dict) and "array" in audio_data:
                         # {'array': ..., 'sampling_rate': ...}
                         waveform = torch.tensor(audio_data["array"]).float()
                         sr = audio_data["sampling_rate"]
                         if sr != 16000:
                             resampler = Resample(orig_freq=sr, new_freq=16000)
                             waveform = resampler(waveform)
                         item["audio"] = waveform
                    elif isinstance(audio_data, str):
                         # Path or URL
                         pass # TODO implement streaming audio extraction from URL

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
    
    if "audio" in batch[0]:
        # Audio might have variable lengths. Pad?
        # For simple tower, maybe we want fixed length or padding
        audios = [item["audio"] for item in batch]
        max_audio_len = max(a.shape[0] for a in audios)
        collated["audio"] = torch.stack([
            torch.nn.functional.pad(a, (0, max_audio_len - a.shape[0]))
            for a in audios
        ])
    return collated
