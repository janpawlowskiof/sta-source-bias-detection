import pytorch_lightning as pl
import numpy as np

import evaluate
import torch
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from typing import List, Any


class SlobertaNERLightningModule(pl.LightningModule):
    def __init__(self, all_labels: List[str], pretrained_model: str, lr: float = 0.0001):
        super().__init__()
        self.all_labels = all_labels
        self.model: AutoModelForTokenClassification = AutoModelForTokenClassification.from_pretrained(
            pretrained_model, num_labels=len(self.all_labels)
        )
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.lr: float = lr

    def setup(self, stage):
        super().setup(stage)
        self.seqeval = evaluate.load("seqeval")

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def training_step(self, batch, **kwargs: Any):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        preds = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        metrics = {}
        metrics["train_loss"] = preds.loss
        self.log_dict(metrics)
        return preds.loss

    def validation_step(self, batch, batch_index):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        preds = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        metrics = self.compute_metrics(preds.logits, labels, prefix="dev")
        metrics["dev_loss"] = preds.loss
        self.log_dict(metrics)

    def test_step(self, batch, batch_index):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        preds = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        metrics = self.compute_metrics(preds.logits, labels, prefix="test")
        metrics["test_loss"] = preds.loss
        self.log_dict(metrics)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        input_ids_as_strings = [
            self.tokenizer.convert_ids_to_tokens(single_input_ids)
            for single_input_ids in input_ids
        ]
        predictions = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits
        predictions = predictions.detach().cpu()
        predictions = np.argmax(predictions, axis=2)
        decoded_preds = [
            [
                f"{token.replace('Ġ', '')}:({self.all_labels[value]}/{self.all_labels[single_label] if single_label != -100 else '-100'})"
                for value, single_label, token
                in zip(single_prediciton, single_labels, single_id_as_text, strict=False)
                if token != "<pad>"
            ]
            for single_prediciton, single_labels, single_id_as_text
            in zip(predictions, labels, input_ids_as_strings, strict=True)
        ]
        return decoded_preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def compute_metrics(self, predictions, labels, prefix: str):
        predictions = predictions.detach().cpu()
        labels = labels.detach().cpu()
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.all_labels[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.seqeval.compute(predictions=true_predictions, references=true_labels, zero_division=0)
        return {
            f"{prefix}_precision": results["overall_precision"],
            f"{prefix}_recall": results["overall_recall"],
            f"{prefix}_f1": results["overall_f1"],
            f"{prefix}_accuracy": results["overall_accuracy"],
        }
