import json
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple
import torch

import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import PreTrainedTokenizer

from app.ner.json_dataset import JsonDataset


class JsonDataModule(pl.LightningDataModule):
    def __init__(self, path: Path, batch_size: int, tokenizer: PreTrainedTokenizer, max_margin_size: int, max_length: int):
        super().__init__()
        self.path: Path = Path(path)
        self.batch_size: int = batch_size
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_margin_size: int = max_margin_size
        self.max_length: int = max_length

    def train_dataloader(self):
        dataset = JsonDataset(self.path, split="train", tokenizer=self.tokenizer, max_margin_size=self.max_margin_size, max_length=self.max_length)
        return DataLoader(dataset, shuffle=True, num_workers=16, batch_size=self.batch_size)

    def val_dataloader(self):
        dataset = JsonDataset(self.path, split="dev", tokenizer=self.tokenizer, max_margin_size=self.max_margin_size, max_length=self.max_length)
        return DataLoader(dataset, shuffle=False, num_workers=16, batch_size=self.batch_size)

    def test_dataloader(self):
        dataset = JsonDataset(self.path, split="test", tokenizer=self.tokenizer, max_margin_size=self.max_margin_size, max_length=self.max_length)
        return DataLoader(dataset, shuffle=False, num_workers=16, batch_size=self.batch_size)

    def predict_dataloader(self):
        return self.test_dataloader()
