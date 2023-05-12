import pytorch_lightning as pl
import numpy as np

import evaluate
import torch
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from typing import List, Any


class SlobertaNERLightningModule(pl.LightningModule):
    def __init__(self, pretrained_model: str, lr: float = 0.0001):
        super().__init__()
        self.save_hyperparameters()

        self.all_labels = ["O", "B-SourceOrg", "I-SourceOrg", "B-SourcePer", "I-SourcePer", "B-NonSource", "I-NonSource"]
        id2label = dict(enumerate(self.all_labels))
        label2id = {v: k for k, v in id2label.items()}

        self.model: AutoModelForTokenClassification = AutoModelForTokenClassification.from_pretrained(
            pretrained_model, id2label=id2label, label2id=label2id,
        )
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.lr: float = lr

    def setup(self, stage):
        super().setup(stage)
        self.seqeval = evaluate.load("seqeval")

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def predict(self, text: str):
        tokenized_input = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids=tokenized_input["input_ids"]
        model_output = self.model(
            input_ids=input_ids, attention_mask=tokenized_input["attention_mask"]
        )
        [input_ids] = input_ids
        [prediction_ids] = torch.argmax(model_output.logits, dim=2)

        entry_results = []
        
        current_entity_ids = []
        current_entity_class = None
        for input_id, prediction_id in zip(input_ids, prediction_ids, strict=True):
            label = self.all_labels[prediction_id]
            if label.startswith("I-") and current_entity_ids:
                current_entity_ids.append(input_id)
            elif label == "O" or label.startswith("B-"):
                if current_entity_ids:
                    entity_text = self.tokenizer.decode(current_entity_ids)
                    entry_results.append((current_entity_class, entity_text))
                    current_entity_ids = []
                    current_entity_class = None
                if label.startswith("B-"):
                    current_entity_class = label.removeprefix("B-")
                    current_entity_ids.append(input_id)
            else:
                raise RuntimeError(f"Found uknown label {label} when decoding")
        if current_entity_ids:
            entry_results.append(self.tokenizer.decode(current_entity_ids))

        return entry_results

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
