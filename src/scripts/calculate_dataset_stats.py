from torch.utils.data import DataLoader
from config.model_config import OCRModelConfig
from utils.natural_dataset import get_natural_datasets
from config.model_config import OCRModelConfig
from config.train_config import OCRTrainConfig


def compute_dataset_stats(
    model_config: OCRModelConfig,
    train_config: OCRTrainConfig,
    dataset_dir,
    preprocessed=False,
):
    train_dataset, _, _ = get_natural_datasets(model_config, dataset_dir, preprocessed)
    loader = DataLoader(
        train_dataset,
        batch_size=train_config.natural_batch_size,
        num_workers=train_config.workers,
    )
    mean, std = 0.0, 0.0
    count = 0
    for img, _, _ in loader:
        batch_size = img.size(0)
        mean += img.mean(dim=[0, 2, 3]).sum()
        std += img.std(dim=[0, 2, 3]).sum()
        count += batch_size
    mean /= count
    std /= count
    return mean.item(), std.item()


model_config = OCRModelConfig()
train_config = OCRTrainConfig()
mean, std = compute_dataset_stats(
    model_config, train_config, "data/raw", preprocessed=False
)
print(f"Grayscale Mean: {mean:.3f}, Std: {std:.3f}")
