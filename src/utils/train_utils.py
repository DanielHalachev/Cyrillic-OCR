import os
import torch
from tqdm import tqdm
from torch.optim import SGD
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
from utils.label_text_mapping import labels_to_text
from utils.synth_dataset import get_synthetic_datasets
from torch.utils.data import DataLoader
from utils.test_utils import test


def get_scheduler(optimizer, total_steps, warmup_steps=5000):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / warmup_steps
        return 1.0

    warmup = LambdaLR(optimizer, lr_lambda)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    return torch.optim.lr_scheduler.ChainedScheduler([warmup, cosine])


def train_epoch(
    model: OCRModel, optimizer, scheduler, criterion, dataloader: DataLoader
):
    model.train()
    training_loss = 0.0
    counter = 0
    scaler = GradScaler()
    for src, _, trg in tqdm(dataloader):
        counter += 1
        if torch.cuda.is_available():
            src, trg = src.cuda(), trg.cuda()
        optimizer.zero_grad()
        with autocast("cuda"):
            output = model(src, trg[:-1])
            loss = criterion(output.view(-1, output.shape[-1]), trg[1:, :].view(-1))
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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


def validate_epoch(wrapper: OCRModelWrapper, criterion, dataloader: DataLoader):
    cer_overall = 0.0
    wer_overall = 0.0
    validation_loss = 0.0
    counter = 0
    wrapper.model.eval()

    with torch.no_grad():
        for src, labels, trg in tqdm(dataloader):
            counter += len(labels)  # Batch size
            if torch.cuda.is_available():
                src = src.cuda()
                trg = trg.cuda()

            validation_loss += calculate_validation_loss(
                wrapper.model, criterion, src, trg
            ) * len(labels)

            predictions = wrapper.predict(src)  # List of predicted texts
            for pred, label in zip(predictions, labels):
                cer_overall += char_error_rate(label, pred) if pred else 1
                wer_overall += word_error_rate(label, pred) if pred else 1

    return (
        cer_overall / counter,
        wer_overall / counter,
        validation_loss / counter,
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
    best_cer = float("inf")
    patience_counter = 0
    for epoch in range(epochs):
        train_loss = train_epoch(
            wrapper.model, optimizer, scheduler, criterion, train_loader
        )
        cer_loss, wer_loss, validation_loss = validate_epoch(
            wrapper, criterion, validation_loader
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
        print(
            f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={validation_loss:.4f}, CER={cer_loss:.4f}, WER={wer_loss:.4f}"
        )
        if cer_loss < best_cer:
            best_cer = cer_loss
            patience_counter = 0
            wrapper.model.save_model(save_path)
            wandb.save(str(Path(os.path.expanduser(save_path))))
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break


def train_synthetic(
    epochs: int,
    config: OCRModelConfig,
    wrapper: OCRModelWrapper,
    batch_size,
    lr,
    weight_decay,
    save_path: os.PathLike,
):
    train_dataset, val_dataset, test_dataset = get_synthetic_datasets(config)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=TextCollate()
    )
    validation_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=TextCollate()
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=TextCollate()
    )

    optimizer = torch.optim.AdamW(
        wrapper.model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss(ignore_index=config.char2idx["PAD"])
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
    wrapper.model.save_model(Path(os.path.expanduser(save_path)))


def train_natural(
    epochs,
    config: OCRModelConfig,
    wrapper: OCRModelWrapper,
    batch_size,
    lr,
    weight_decay,
    dataset_dir: os.PathLike,
    save_path: os.PathLike,
):
    train_dataset, val_dataset, test_dataset = get_natural_datasets(config, dataset_dir)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=TextCollate()
    )
    validation_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=TextCollate()
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=TextCollate()
    )

    optimizer = torch.optim.AdamW(
        wrapper.model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss(ignore_index=config.char2idx["PAD"])
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

    wrapper.model.save_model(Path(os.path.expanduser(save_path)))
