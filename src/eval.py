import argparse

import lightning as L
import torch

from datamodules.dog_breed_datamodule import DogBreedDataModule
from models.dog_breed_classifier import DogBreedClassifier
from utils.logging_utils import setup_logger, task_wrapper
from loguru import logger

# Set up the logger
setup_logger("eval.log")

@task_wrapper
def main(args):

    # 1. data module
    data_module = DogBreedDataModule(num_workers=2, batch_size=16)

    # 2. set up the data module for validation data
    data_module.setup(stage="fit")

    # 3. validation datset
    val_dataset = data_module.val_dataset

    # 4. data loader
    val_data_loader = data_module.val_dataloader()

    # 5. load model
    model = DogBreedClassifier.load_from_checkpoint(args.ckpt_path)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # 6. Trainer
    trainer = L.Trainer(
        max_epochs=1,
        log_every_n_steps=10,
        accelerator="auto",
    )

    # 7. evaluate the module
    results = trainer.test(
        model=model,
        datamodule=data_module,
    )

    logger.info("Validation is completed!!!")
    logger.info(f"validation results: {results}")


if __name__ == "__main__":

    """
    This function parses command line arguments for the model checkpoint path and calls the main function to perform evaluation on images.

    """

    parser = argparse.ArgumentParser(description="Performs evaluation on images")

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="path to the model checkpoint",
    )

    args = parser.parse_args()
    main(args)