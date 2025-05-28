from argparse import ArgumentParser
import os
from pathlib import Path

import torch

from backbones.cnn import CNNBackbone
from config.model_config import OCRModelConfig
from config.train_config import OCRTrainConfig
from models.ocr_model import OCRModel, ResnetOCRModel
from models.ocr_model_wrapper import OCRModelWrapper
from utils.test_utils import test
from utils.train_utils import train_natural, train_synthetic

import wandb

if __name__ == "__main__":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/",
        help="Directory containing training images",
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model_config = OCRModelConfig()
    train_config = OCRTrainConfig()
    # model = ResnetOCRModel(model_config)
    # model = model.to(device)
    # print("Device: ", next(model.parameters()).device)
    # wrapper = OCRModelWrapper(model, model_config, device)

    # with wandb.init(project="cyrillic-ocr-synthetic"):
    #     wandb.config.update(
    #         {
    #             "learning_rate": train_config.synthetic_lr,
    #             "epochs": train_config.synthetic_epochs,
    #             "batch_size": train_config.synthetic_batch_size,
    #             "dataset": "https://huggingface.co/datasets/pumb-ai/synthetic-cyrillic-large",
    #         }
    #     )

    #     wandb.watch(model)

    #     train_synthetic(
    #         train_config.synthetic_epochs,
    #         model_config,
    #         wrapper,
    #         train_config.synthetic_batch_size,
    #         train_config.synthetic_lr,
    #         train_config.synthetic_decay_rate,
    #         Path(train_config.synthetic_checkpoint_path),
    #         train_config.workers,
    #     )

    #     wandb.finish()

    # if everything goes well with pretraining, uncomment for fine-tuning
    with wandb.init(project="cyrillic-ocr-natural"):
        wandb.config.update(
            {
                "learning_rate": train_config.natural_lr,
                "epochs": train_config.natural_epochs,
                "batch_size": train_config.natural_batch_size,
                "dataset": "https://www.kaggle.com/datasets/constantinwerner/cyrillic-handwriting-dataset",
            }
        )

        model = OCRModel.load_from_checkpoint(
            model_path=Path(train_config.synthetic_checkpoint_path),
            device="cuda",
            backbone=CNNBackbone(model_config.hidden, pretrained=True),
            outtoken=len(model_config.tokens),
            hidden=model_config.hidden,
            enc_layers=model_config.enc_layers,
            dec_layers=model_config.dec_layers,
            nhead=model_config.nhead,
            dropout=model_config.dropout,
        )
        wrapper = OCRModelWrapper(model, model_config, device)
        print("Device: ", next(model.parameters()).device)

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
            train_config.workers,
            train_config.preprocessed,
        )

        wandb.finish()
