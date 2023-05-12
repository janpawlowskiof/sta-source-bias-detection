from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from app import ROOT_PATH

from app.ner.lightning_module import SlobertaNERLightningModule


class SourceExtractor:
    def __init__(self, model: AutoModelForTokenClassification, tokenizer: AutoTokenizer) -> None:
        self.pipeline = pipeline(
            "token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="first"
        )

    @classmethod
    def from_wandb_checkpoint(cls, wandb_checkpoint_path) -> SourceExtractor:
        import wandb
        artifact_dir = wandb.Api().artifact(wandb_checkpoint_path).download()
        model_path = Path(artifact_dir) / "model.ckpt"
        model = SlobertaNERLightningModule.load_from_checkpoint(model_path)
        return cls(model=model.model, tokenizer=model.tokenizer)

    @classmethod
    def from_default_model(cls, version: str = "model.ckpt") -> SourceExtractor:
        checkpoint_path = ROOT_PATH / "source_extractor_models" / version
        model = SlobertaNERLightningModule.load_from_checkpoint(checkpoint_path)
        return cls(model=model.model, tokenizer=model.tokenizer)

    def extract_sources_from_texts(self, texts: List[str]) -> List[List[str]]:
        return [
            {
                "text": text,
                "entities": self.extract_sources_from_text(text) 
            }
            for text in tqdm(texts)
        ]

    def extract_sources_from_text(self, text: str) -> List[Dict[str, str]]:
        if len(text) < 10:
            print(f"Text '{text}' has only {len(text)} characters, and will not be processed")
            return []

        entities = [
            {
                "entity_value": text[entity["start"]: entity["end"]],
                "label": entity["entity_group"],
                "start": entity["start"],
                "end": entity["end"],
            }
            for entity in self.pipeline(text)
            if entity["entity_group"] != "NonSource"
        ]
        entities = [
            self.cleanup_entity(entity)
            for entity in entities
        ]
        return entities

    def cleanup_entity(self, entity):
        text: str = entity["entity_value"]
        text_no_prefix = re.sub(r"(^[^\w]+)", "", text)
        text_no_prefix_no_suffix = re.sub(r"([^\w]+$)", "", text_no_prefix)
        entity["entity_value"] = text_no_prefix_no_suffix
        num_removed_prefix_chars = len(text) - len(text_no_prefix)
        entity["start"] += num_removed_prefix_chars
        num_removed_suffix_chars = len(text_no_prefix) - len(text_no_prefix_no_suffix)
        entity["end"] -= num_removed_suffix_chars
        return entity

    def split_text(self, text) -> List[str]:
        return [str(sent) for sent in self.sentencizer(text).sents]
