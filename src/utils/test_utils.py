import torch
import re
from tqdm import tqdm
from models.ocr_model import OCRModel
from models.ocr_model_wrapper import OCRModelWrapper
from utils.error_rates import char_error_rate, word_error_rate
from torch.utils.data import DataLoader


def test(
    wrapper: OCRModelWrapper,
    dataloader: DataLoader,
    case=True,
    punct=False,
):
    cer_overall = 0.0
    wer_overall = 0.0
    counter = 0
    wrapper.model.eval()

    with torch.no_grad():
        for src, labels, trg in tqdm(dataloader):
            counter += len(labels)  # Batch size
            if torch.cuda.is_available():
                src = src.cuda()
                trg = trg.cuda()

            predictions = wrapper.predict(src)  # List of predicted texts
            for pred, label in zip(predictions, labels):
                # Process prediction and label based on case and punct settings
                processed_pred = process_text(pred, case, punct) if pred else ""
                processed_label = process_text(label, case, punct)

                cer_overall += (
                    char_error_rate(processed_label, processed_pred)
                    if processed_pred
                    else 1
                )
                wer_overall += (
                    word_error_rate(processed_label, processed_pred)
                    if processed_pred
                    else 1
                )

    return (
        cer_overall / counter,
        wer_overall / counter,
    )


def process_text(text, preserve_case=True, preserve_punct=True):
    """
    Process text based on case and punctuation preferences.
    """
    if not preserve_case:
        text = text.lower()

    if not preserve_punct:
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)

    return text
