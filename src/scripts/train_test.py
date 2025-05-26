from argparse import ArgumentParser
import os
from pathlib import Path

import torch

from config.model_config import OCRModelConfig
from config.train_config import OCRTrainConfig
from models.ocr_model import ResnetOCRModel
from models.ocr_model_wrapper import OCRModelWrapper
from utils.test_utils import test
from utils.train_utils import train_natural, train_synthetic

import wandb

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/",
        help="Directory containing training images",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = OCRModelConfig()
    train_config = OCRTrainConfig()
    model = ResnetOCRModel(model_config)
    wrapper = OCRModelWrapper(model, model_config, device)

    # with wandb.init(project="cyrillic-ocr-synthetic"):
    #     wandb.config.update(
    #         {
    #             "learning_rate": train_config.synthentic_lr,
    #             "epochs": train_config.synthetic_epochs,
    #             "batch_size": train_config.synthetic_batch_size,
    #         }
    #     )

    #     wandb.watch(model)

    #     train_synthetic(
    #         train_config.synthetic_epochs,
    #         model_config,
    #         wrapper,
    #         train_config.synthetic_batch_size,
    #         train_config.synthentic_lr,
    #         train_config.synthetic_decay_rate,
    #         Path(train_config.synthetic_checkpoint_path),
    #     )

    #     wandb.finish()

    # if everything goes well with pretraining, uncomment for fine-tuning
    with wandb.init(project="cyrillic-ocr-natural"):
        wandb.config.update(
            {
                "learning_rate": train_config.natural_lr,
                "epochs": train_config.natural_epochs,
                "batch_size": train_config.natural_batch_size,
            }
        )
        # x
        wrapper = OCRModelWrapper(model, model_config, device)
        wandb.watch(model)
        train_natural(
            train_config.natural_epochs,
            model_config,
            wrapper,
            train_config.natural_batch_size,
            train_config.natural_lr,
            train_config.natural_decay_rate,
            args.dataset_dir,
            Path(train_config.natural_checkpoint_path),
        )

        wandb.finish()
