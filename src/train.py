import os
from pathlib import Path
import argparse

import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger

from datamodules.dog_breed_datamodule import DogBreedDataModule
from models.dog_breed_classifier import DogBreedClassifier
from utils.logging_utils import setup_logger, task_wrapper


@task_wrapper
def train_and_test(
    data_module: DogBreedDataModule, model: DogBreedClassifier, trainer: L.Trainer
):
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="logs/dog_breed_classification/checkpoints/best_model.ckpt",
    )
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    args = parser.parse_args()

    # Set up paths
    root_dir = Path(__file__).parent.parent
    data_dir = root_dir / "data"
    log_dir = root_dir / "logs"

    # Set up loggers
    setup_logger(log_dir / "train_log.log")

    # Set up data module
    data_module = DogBreedDataModule(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    data_module.prepare_data()
    data_module.setup(stage="fit")

    # Get the number of classes
    num_classes = len(data_module.train_dataset.dataset.classes)

    # Set up model
    if args.load_model:
        model = DogBreedClassifier.load_from_checkpoint(args.load_model)
    else:
        model = DogBreedClassifier(
            num_classes=num_classes,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir / "dog_breed_classification" / "checkpoints",
        filename="best_model",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    early_stopping_callback = EarlyStopping(monitor="val_acc", patience=5, mode="max")
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Initialize Trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            RichProgressBar(),
            RichModelSummary(max_depth=2),
            lr_monitor,
        ],
        accelerator="auto",
        devices="auto",
        logger=TensorBoardLogger(save_dir=log_dir, name="dog_breed_classification"),
        val_check_interval=0.5,  # Validate twice per epoch
        precision="16-mixed",
        log_every_n_steps=10,
    )

    # Train and test the model
    train_and_test(data_module, model, trainer)


if __name__ == "__main__":
    main()
