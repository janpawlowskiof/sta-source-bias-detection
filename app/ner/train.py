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
from app.ner.lightning_module import SlobertaNERLightningModule


def main():
    config_path = Path(__file__).parent / "config.yaml"
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    train_model(config)


def train_model(config: Dict):
    sloberta_module = SlobertaNERLightningModule(
        pretrained_model=config["pretrained_model"], 
        lr=config["lr"], 
    )
    
    wandb_run = WandbLogger(
        config=config,
        project=config["wandb_project"],
        group=f"ner_{config['model_name']}",
        log_model=True,
        resume="allow",
        reinit=True,
    )

    model_path: Path = Path(config["models_path"]) / config["model_name"]
    checkpoint_callback: ModelCheckpoint = ModelCheckpoint(
        dirpath=model_path, monitor="dev_loss", mode="min", save_last=True
    )

    data_module = JsonDataModule(
        path=config["data_path"], 
        batch_size=config["batch_size"], 
        tokenizer=sloberta_module.tokenizer, 
        max_length=config["max_length"],
        max_margin_size=config["max_margin_size"],
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


if __name__ == "__main__":
    main()
