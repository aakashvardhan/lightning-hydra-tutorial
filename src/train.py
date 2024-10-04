import os
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from datamodules.dog_breed_datamodule import DogBreedDataModule
from models.dog_breed_classifier import DogBreedClassifier
from utils.logging_utils import setup_logger, task_wrapper

@task_wrapper
def train_and_test(data_module: DogBreedDataModule, model: DogBreedClassifier, trainer: L.Trainer):
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

def main():
    # Set up paths
    root_dir = Path(__file__).parent.parent
    data_dir = root_dir / "data"
    log_dir = root_dir / "logs"
    
    # Set up loggers
    setup_logger(log_dir / "train_log.log")
    
    # Set up data module with more workers
    data_module = DogBreedDataModule(data_dir=data_dir, batch_size=16, num_workers=0)  # Reduced batch size to 16
    data_module.prepare_data()
    data_module.setup(stage="fit")
    
    # Get the number of classes
    num_classes = len(data_module.train_dataset.dataset.classes)
    
    # Set up model with adjusted parameters
    model = DogBreedClassifier(num_classes=num_classes, lr=5e-4, weight_decay=1e-4)
    
    # Set up checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir / "dog_breed_classification" / "checkpoints",
        filename="epoch={epoch:02d}-val_acc={val_acc:.2f}",
        save_top_k=3,
        monitor="val_acc",
        mode="max",
    )

    # Add early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor="val_acc",
        patience=5,
        mode="max"
    )

    # Initialize Trainer with updated parameters
    trainer = L.Trainer(
        max_epochs=5,  # Increase max epochs
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            RichProgressBar(),
            RichModelSummary(max_depth=2)
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