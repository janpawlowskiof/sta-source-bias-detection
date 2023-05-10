from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
import torch


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
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/sloberta")
        self.model = AutoModelForSequenceClassification.from_pretrained("EMBEDDIA/sloberta")
        self.model.to('cuda')

    def train(self, out_dir: str = './saved_model/results'):
        train_dataset = self._load_data(self.data_file)

        training_args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./saved_model/logs',
            logging_steps=10,
            save_strategy="epoch",
            overwrite_output_dir=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )

        trainer.train()

    def _load_data(self, data_file):
        ds = pd.read_csv(data_file)

        train_texts = list(ds['text'])
        train_labels = list(ds['label'])

        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True)
        return AttributionDataset(train_encodings, train_labels)
