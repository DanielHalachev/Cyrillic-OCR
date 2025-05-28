from datasets import load_dataset  # type: ignore
from torchvision import transforms  # type: ignore
from PIL import Image
import torch
from torch.utils.data import Subset
import numpy as np

from config.model_config import OCRModelConfig
from utils.label_text_mapping import text_to_labels
from utils.resize_and_pad import ResizeAndPadTransform


class SyntheticCyrillicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_split, config: OCRModelConfig, eval=False):
        self.dataset = dataset_split
        self.config = config
        self.eval = eval

        self.transform = transforms.Compose(
            [
                # transforms.Grayscale(num_output_channels=1),  # for 1-dim input
                transforms.Grayscale(num_output_channels=3),  # for 3-dim input
                ResizeAndPadTransform(self.config.height, self.config.width),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.natural_mean,
                    std=self.config.natural_std,
                ),
            ]
            if eval
            else [
                # transforms.Grayscale(num_output_channels=1), # for 1-dim input
                transforms.Grayscale(num_output_channels=3),  # for 3-dim input
                ResizeAndPadTransform(self.config.height, self.config.width),
                # transforms.ColorJitter(contrast=(0.5, 1)),
                transforms.RandomRotation(degrees=(-9, 9), fill=255),
                transforms.RandomAffine(degrees=5, scale=(0.9, 1.1), shear=2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.natural_mean,
                    std=self.config.natural_std,
                ),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        img_tensor = self.transform(item["png"])
        text = item["txt"]
        label_encoding = text_to_labels(text, self.config)
        return img_tensor, text, torch.LongTensor(label_encoding)


def get_synthetic_datasets(config: OCRModelConfig):
    fraction = 0.25
    full_dataset = load_dataset("pumb-ai/synthetic-cyrillic-large")

    trainval_data = full_dataset["train"]
    test_data = full_dataset["test"]

    total_size = len(trainval_data)
    reduced_size = int(total_size * fraction)
    indices = list(range(total_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    indices = indices[:reduced_size]

    train_end = int(0.9474 * total_size)  # ~90% of entire dataset
    train_indices = indices[:train_end]
    val_indices = indices[train_end:]

    train_split = Subset(trainval_data, train_indices)
    val_split = Subset(trainval_data, val_indices)

    test_size = len(test_data)
    reduced_test_size = int(test_size * fraction)
    test_indices = list(range(test_size))
    np.random.seed(42)
    np.random.shuffle(test_indices)
    test_indices = test_indices[:reduced_test_size]

    test_split = Subset(test_data, test_indices)

    # Create datasets
    train_dataset = SyntheticCyrillicDataset(train_split, config, eval=False)
    val_dataset = SyntheticCyrillicDataset(val_split, config, eval=True)
    test_dataset = SyntheticCyrillicDataset(test_split, config, eval=True)

    return train_dataset, val_dataset, test_dataset
