from pathlib import Path
from typing import Dict

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
import wandb
import yaml

from app.ner.data_module import JsonDataModule
from app.ner.models import SlobertaNERLightningModule


def main():
    config_path = Path(__file__).parent / "config.yaml"
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    train_model(config)


def train_model(config: Dict):
    all_labels = ["O", "B-entity", "I-entity"]

    sloberta_module = SlobertaNERLightningModule(
        pretrained_model=config["pretrained_model"], 
        lr=config["lr"], 
        all_labels=all_labels
    )
    data_module = JsonDataModule(
        path=config["data_path"], 
        batch_size=config["batch_size"], 
        tokenizer=sloberta_module.tokenizer, 
        all_labels=all_labels,
        max_margin_size=config["max_margin_size"],
        max_length=config["max_length"]
    )

    model_path: Path = Path(config["models_path"]) / config["model_name"]
    checkpoint_callback: ModelCheckpoint = ModelCheckpoint(
        dirpath=model_path, monitor="dev_loss", mode="min", save_last=True
    )

    wandb_run = WandbLogger(
        config=config,
        project=config["wandb_project"],
        group=f"ner_{config['model_name']}",
        log_model=True,
        resume="allow",
        reinit=True,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=(config["gpu_id"],) if torch.cuda.is_available() else None,
        max_epochs=config["epochs"],
        callbacks=[TQDMProgressBar(), checkpoint_callback],
        logger=wandb_run,
    )

    trainer.fit(sloberta_module, datamodule=data_module)
    trainer.test(sloberta_module, datamodule=data_module)
    # predictions = trainer.predict(sloberta_module, datamodule=data_module)
    # predictions = sum(predictions, [])

    # artifact = wandb.Artifact("predictions", type="predictions", incremental=False)
    # df = pd.DataFrame(predictions)
    # table = wandb.Table(dataframe=df, dtype=str)
    # artifact.add(table, name="predictions")
    # wandb_run._experiment.log_artifact(artifact)

    # correct_count, wrong_count = calculate_predictions_order(predictions)
    # artifact = wandb.Artifact("predictions_order", type="predictions", incremental=False)
    # df = pd.DataFrame({"correct": [correct_count], "wrong": [wrong_count]})
    # table = wandb.Table(dataframe=df, dtype=int)
    # artifact.add(table, name="predictions_order")
    # wandb_run._experiment.log_artifact(artifact)


if __name__ == "__main__":
    main()
