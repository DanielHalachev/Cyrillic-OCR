import os
import torch
from tqdm import tqdm
from config.model_config import OCRModelConfig
from models.ocr_model import OCRModel
import torch.nn as nn
from pathlib import Path
import wandb
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from models.ocr_model_wrapper import OCRModelWrapper
from utils.collate import TextCollate
from utils.natural_dataset import get_natural_datasets
from utils.error_rates import char_error_rate, word_error_rate
from utils.synth_dataset import get_synthetic_datasets
from torch.utils.data import DataLoader
from utils.test_utils import test
import random
import torchvision.transforms  # type:ignore


def get_scheduler(optimizer, total_steps, warmup_steps=5000):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / warmup_steps
        return 1.0

    warmup = LambdaLR(optimizer, lr_lambda)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    return torch.optim.lr_scheduler.ChainedScheduler([warmup, cosine])


def train_epoch(
    wrapper: OCRModelWrapper, optimizer, scheduler, criterion, dataloader: DataLoader
):
    wrapper.model.train()
    training_loss = 0.0
    counter = 0
    scaler = GradScaler()
    for src, _, trg in tqdm(dataloader):
        src = src.to(wrapper.device)
        trg = trg.to(wrapper.device)

        counter += 1

        optimizer.zero_grad()
        with autocast("mps"):
            output = wrapper.model(src, trg[:-1])
            loss = criterion(output.view(-1, output.shape[-1]), trg[1:, :].view(-1))
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(wrapper.model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        training_loss += loss.item()
    return training_loss / counter


def calculate_validation_loss(model: OCRModel, criterion, src, trg):
    output = model(src, trg[:-1, :])
    loss = criterion(
        output.view(-1, output.shape[-1]), torch.reshape(trg[1:, :], (-1,))
    )
    return loss.item()


def validate_epoch(
    epoch: int,
    wrapper: OCRModelWrapper,
    criterion,
    dataloader: DataLoader,
    num_samples_to_log=5,
):
    cer_overall = 0.0
    wer_overall = 0.0
    validation_loss = 0.0
    counter = 0
    wrapper.model.eval()

    logged_samples = 0
    total_samples = 0

    records = []

    cfg = OCRModelConfig()
    IMG_MEAN = cfg.natural_mean
    IMG_STD = cfg.natural_std

    with torch.no_grad():
        for src, labels, trg in tqdm(dataloader):
            src = src.to(wrapper.device)  # [B, C, H, W]
            trg = trg.to(wrapper.device)  # [L, B]

            batch_size = len(labels)
            counter += batch_size
            total_samples += batch_size

            validation_loss += (
                calculate_validation_loss(wrapper.model, criterion, src, trg)
                * batch_size
            )

            predictions = wrapper.predict(src)  # List of predicted texts
            for i, (pred, label) in enumerate(zip(predictions, labels)):
                cer = char_error_rate(label, pred) if pred else 1.0
                wer = word_error_rate(label, pred) if pred else 1.0
                cer_overall += cer
                wer_overall += wer

                # Randomly select samples to log
                if logged_samples < num_samples_to_log and random.random() < (
                    num_samples_to_log / max(total_samples, 1)
                ):
                    # Un-normalize and convert image tensor to PIL
                    img = src[i].cpu()  # [C, H, W]
                    # for 1-dim input
                    # if img.shape[0] != 1:
                    #     raise ValueError(
                    #         f"Expected grayscale image with 1 channel, got {img.shape[0]} channels"
                    #     )
                    IMG_MEAN = torch.tensor(IMG_MEAN).view(3, 1, 1)
                    IMG_STD = torch.tensor(IMG_STD).view(3, 1, 1)
                    img = img * IMG_STD + IMG_MEAN
                    img = img.clamp(0, 1)
                    # img = img.squeeze(0)  # [H, W]
                    img = torchvision.transforms.ToPILImage()(img)
                    records.append((epoch, img, label, pred, cer, wer))
                    logged_samples += 1

    return (
        cer_overall / counter,
        wer_overall / counter,
        validation_loss / counter,
        records,
    )


def train(
    wrapper: OCRModelWrapper,
    optimizer,
    criterion,
    scheduler,
    train_loader,
    validation_loader,
    epochs,
    save_path,
    patience=5,
):
    columns = ["Epoch", "Image", "Ground Truth", "Prediction", "CER", "WER"]
    all_records = []
    best_cer = float("inf")
    patience_counter = 0
    for epoch in range(epochs):
        print(f"==================== EPOCH {(epoch + 1)} ====================")
        train_loss = train_epoch(wrapper, optimizer, scheduler, criterion, train_loader)
        cer_loss, wer_loss, validation_loss, records = validate_epoch(
            epoch, wrapper, criterion, validation_loader, num_samples_to_log=5
        )
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "cer_loss": cer_loss,
                "wer_loss": wer_loss,
            }
        )

        all_records.extend(
            [
                (epoch, img, label, pred, cer, wer)
                for epoch, img, label, pred, cer, wer in records
            ]
        )
        table = wandb.Table(columns=columns, data=all_records)
        wandb.log({"validation_samples": table})

        print(
            f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={validation_loss:.4f}, CER={cer_loss:.4f}, WER={wer_loss:.4f}"
        )
        if cer_loss < best_cer:
            best_cer = cer_loss
            patience_counter = 0
            wrapper.model.save_model(save_path)
            wandb.save(str(Path(save_path)))
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break


def train_synthetic(
    epochs: int,
    cfg: OCRModelConfig,
    wrapper: OCRModelWrapper,
    batch_size,
    lr,
    weight_decay,
    save_path: os.PathLike,
    workers,
):
    train_dataset, val_dataset, test_dataset = get_synthetic_datasets(cfg)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(False if wrapper.device.type == "mps" else True),
        num_workers=workers,
        collate_fn=TextCollate(),
    )
    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(False if wrapper.device.type == "mps" else True),
        num_workers=workers,
        collate_fn=TextCollate(),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(False if wrapper.device.type == "mps" else True),
        num_workers=workers,
        collate_fn=TextCollate(),
    )

    optimizer = torch.optim.AdamW(
        wrapper.model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.char2idx["PAD"])
    scheduler = get_scheduler(optimizer, total_steps=(epochs * len(train_loader)) // 10)

    train(
        wrapper,
        optimizer,
        criterion,
        scheduler,
        train_loader,
        validation_loader,
        epochs,
        save_path,
    )

    test_cer, test_wer = test(wrapper, test_loader)
    wandb.log(
        {
            "test_cer": test_cer,
            "test_wer": test_wer,
        }
    )
    wrapper.model.save_model(Path(save_path))


def train_natural(
    epochs,
    cfg: OCRModelConfig,
    wrapper: OCRModelWrapper,
    batch_size,
    lr,
    weight_decay,
    dataset_dir: os.PathLike,
    save_path: os.PathLike,
    workers,
    preprocessed=False,
):
    train_dataset, val_dataset, test_dataset = get_natural_datasets(
        cfg, dataset_dir, preprocessed
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(False if wrapper.device.type == "mps" else True),
        num_workers=workers,
        collate_fn=TextCollate(),
    )
    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(False if wrapper.device.type == "mps" else True),
        num_workers=workers,
        collate_fn=TextCollate(),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(False if wrapper.device.type == "mps" else True),
        num_workers=workers,
        collate_fn=TextCollate(),
    )

    optimizer = torch.optim.AdamW(
        wrapper.model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.char2idx["PAD"])
    scheduler = get_scheduler(optimizer, total_steps=(epochs * len(train_loader)) // 10)

    train(
        wrapper,
        optimizer,
        criterion,
        scheduler,
        train_loader,
        validation_loader,
        epochs,
        save_path,
    )

    test(wrapper, test_loader)

    wrapper.model.save_model(Path(save_path))
