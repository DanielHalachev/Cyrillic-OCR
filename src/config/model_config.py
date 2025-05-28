import json
import os
from pathlib import Path

from config.backbone_type import BackboneType

TOKENS = [
    "PAD",
    "SOS",
    " ",
    "!",
    '"',
    "%",
    "(",
    ")",
    ",",
    "-",
    ".",
    "/",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    ";",
    "?",
    "[",
    "]",
    "«",
    "»",
    "„",
    "“",
    "А",
    "Б",
    "В",
    "Г",
    "Д",
    "Е",
    "Ж",
    "З",
    "И",
    "Й",
    "К",
    "Л",
    "М",
    "Н",
    "О",
    "П",
    "Р",
    "С",
    "Т",
    "У",
    "Ф",
    "Х",
    "Ц",
    "Ч",
    "Ш",
    "Щ",
    "Ъ",
    "Э",
    "Ю",
    "Я",
    "а",
    "б",
    "в",
    "г",
    "д",
    "е",
    "ж",
    "з",
    "и",
    "й",
    "к",
    "л",
    "м",
    "н",
    "о",
    "п",
    "р",
    "с",
    "т",
    "у",
    "ф",
    "х",
    "ц",
    "ч",
    "ш",
    "щ",
    "ъ",
    "ы",
    "ь",
    "э",
    "ю",
    "я",
    "ё",
    "EOS",
]


class OCRModelConfig:
    """
    Attributes:
        tokens (list): List of Cyrillic characters.
        del_sym (list): List of characters to delete.
        hidden (int): Number of hidden units of the Transformer.
        enc_layers (int): Number of encoder layers.
        dec_layers (int): Number of decoder layers.
        nhead (int): Number of heads in the multihead attention models.
        dropout (float): Dropout rate.
        width (int): Width of the input image.
        height (int): Height of the input image.
    """

    def __init__(self, config_path=None):
        """
        Initialize hyperparameters.

        Args:
            config_path (str, optional): Path to the JSON configuration file.
        """
        self.tokens = TOKENS
        self.char2idx = {char: idx for idx, char in enumerate(self.tokens)}
        self.idx2char = {idx: char for idx, char in enumerate(self.tokens)}
        self.non_char_tokens = ["SOS", "EOS", "PAD"]
        self.backbone_type = BackboneType.RESNET_50
        self.del_sym = []
        self.hidden = 512
        self.enc_layers = 5
        self.dec_layers = 4
        self.nhead = 8
        self.dropout = 0.1
        self.width = 256
        self.height = 64
        self.max_length = 100
        self.natural_mean = [0.7564554810523987, 0.7564554810523987, 0.7564554810523987]
        self.natural_std = [0.2374454289674759, 0.2374454289674759, 0.2374454289674759]
        self.synthetic_mean = [
            0.7564554810523987,
            0.7564554810523987,
            0.7564554810523987,
        ]
        self.synthetic_std = [
            0.2374454289674759,
            0.2374454289674759,
            0.2374454289674759,
        ]

        # Load values from JSON file if provided
        if config_path is not None:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            if not os.path.exists(config_path):
                config = vars(self).copy()
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=4)
            else:
                with open(config_path, "r") as f:
                    config = json.load(f)
                for key, value in config.items():
                    setattr(self, key, value)
