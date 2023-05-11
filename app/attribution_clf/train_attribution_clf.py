from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import json
import torch
from datasets import load_metric
import numpy as np


class AttributionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class AttributionClassifierTrainer:
    def __init__(self, train_data_file: str, eval_data_file: str):
        self.train_data_file = train_data_file
        self.eval_data_file = eval_data_file
        self.tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/sloberta")
        self.model = AutoModelForSequenceClassification.from_pretrained("EMBEDDIA/sloberta")
        self.model.to('cuda')
        self.metric = load_metric('f1')

    def train(self, out_dir: str = './saved_model/results_b48'):
        train_dataset = self._load_data(self.train_data_file)
        eval_dataset = self._load_data(self.eval_data_file)

        training_args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=20,
            per_device_train_batch_size=48,
            warmup_steps=50,
            weight_decay=0.001,
            learning_rate=8e-6,
            logging_dir='./saved_model/logs',
            logging_steps=10,
            save_strategy="epoch",
            overwrite_output_dir=True,
            evaluation_strategy='epoch',
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self._compute_metrics
        )

        trainer.train()

    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=labels)

    def _load_data(self, data_file):
        with open(data_file, encoding='utf-8') as f:
            texts = []
            labels = []

            for line in f:
                row = json.loads(line)
                label = row['chatgpt_label']
                if label is not None:
                    texts.append(f"{row['entity']} [SEP] {row['context']}")
                    labels.append(int(bool(label)))

        train_encodings = self.tokenizer(texts, truncation=True, padding=True)
        return AttributionDataset(train_encodings, labels)


if __name__ == '__main__':
    trainer = AttributionClassifierTrainer('../../data/ner_results_labelled_1_5k_train.jsonl',
                                           '../../data/ner_results_labelled_1_5k_test.jsonl')
    trainer.train()
