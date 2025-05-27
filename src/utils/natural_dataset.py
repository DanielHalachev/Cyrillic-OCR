from typing import List, Tuple
import torch
from torchvision import transforms  # type: ignore
from PIL import Image
import os
import random
from config.model_config import OCRModelConfig
from utils.resize_and_pad import resize_and_pad
from utils.label_text_mapping import text_to_labels


def process_data(
    image_dir: str | os.PathLike, labels_file: str | os.PathLike, ignore=[]
) -> List[Tuple[str, str]]:
    img_label_pairs = []
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f.read().splitlines():
            try:
                name, label = line.split("\t")
                if label not in ignore:
                    img_path = os.path.join(image_dir, name)
                    img_label_pairs.append((img_path, label))
            except ValueError:
                continue
    return img_label_pairs


def split_dataset(
    data: List[Tuple[str, str]],
    train_ratio: float = 0.9525,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    random.seed(seed)
    random.shuffle(data)

    total = len(data)
    train_end = int(total * train_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:]

    return train_data, val_data


class ResizeAndPadTransform:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img):
        return resize_and_pad(img, self.height, self.width)


class OCRModelNaturalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: List[Tuple[str, str]],
        config: OCRModelConfig,
        preprocessed: bool,
        eval: bool,
    ):
        self.config = config
        self.image_paths, self.labels = zip(*data)
        self.preprocessed = preprocessed
        self.eval = eval
        if self.preprocessed:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
                if eval
                else [
                    transforms.ColorJitter(contrast=(0.5, 1)),
                    transforms.RandomRotation(degrees=(-9, 9), fill=255),
                    transforms.RandomAffine(degrees=5, scale=(0.9, 1.1), shear=2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=1),
                    ResizeAndPadTransform(self.config.height, self.config.width),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
                if eval
                else [
                    transforms.Grayscale(num_output_channels=1),
                    ResizeAndPadTransform(self.config.height, self.config.width),
                    transforms.ColorJitter(contrast=(0.5, 1)),
                    transforms.RandomRotation(degrees=(-9, 9), fill=255),
                    transforms.RandomAffine(degrees=5, scale=(0.9, 1.1), shear=2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            )

    def __getitem__(self, index):
        img = (
            Image.open(self.image_paths[index]).convert("L")
            if self.preprocessed
            else Image.open(self.image_paths[index]).convert("RGB")
        )

        img_tensor: torch.Tensor = self.transform(img)  # type: ignore
        label_encoding = text_to_labels(self.labels[index], self.config)
        return (
            img_tensor,
            self.labels[index],
            torch.LongTensor(label_encoding),
        )

    def __len__(self):
        return len(self.image_paths)


def get_natural_datasets(
    config: OCRModelConfig, dataset_dir: str | os.PathLike, preprocessed
):
    train_val_data = process_data(
        os.path.join(dataset_dir, "train"), os.path.join(dataset_dir, "train.tsv")
    )
    train_data, val_data = split_dataset(train_val_data, train_ratio=0.9)
    test_data = process_data(
        os.path.join(dataset_dir, "test"), os.path.join(dataset_dir, "test.tsv")
    )

    train_dataset = OCRModelNaturalDataset(
        train_data,
        config,
        preprocessed,
        eval=False,
    )
    val_dataset = OCRModelNaturalDataset(val_data, config, preprocessed, eval=True)
    test_dataset = OCRModelNaturalDataset(test_data, config, preprocessed, eval=True)

    return train_dataset, val_dataset, test_dataset
