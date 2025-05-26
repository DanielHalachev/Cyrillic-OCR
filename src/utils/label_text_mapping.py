from config.model_config import OCRModelConfig
import torch


def labels_to_text(s, config: OCRModelConfig):
    S = "".join(
        [
            (
                config.idx2char[j.item() if isinstance(j, torch.Tensor) else j]  # type: ignore
                if config.idx2char[j.item() if isinstance(j, torch.Tensor) else j]  # type: ignore
                not in config.non_char_tokens
                else ""
            )
            for i in s
            for j in (i.flatten() if isinstance(i, torch.Tensor) else [i])
        ]
    )

    return S.strip()


def text_to_labels(s, config: OCRModelConfig):
    return (
        [config.char2idx["SOS"]]
        + [config.char2idx[i] for i in s if i in config.char2idx.keys()]
        + [config.char2idx["EOS"]]
    )
