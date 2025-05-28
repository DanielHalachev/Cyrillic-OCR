import os
from tqdm import tqdm
import torch
from torchvision import transforms  # type:ignore
from PIL import Image


def compute_mean_std(image_dir):
    transform = transforms.ToTensor()  # Only tensor, no normalization
    pixel_sum = torch.zeros(3)
    pixel_squared_sum = torch.zeros(3)
    num_pixels = 0

    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for file_name in tqdm(image_files):
        img_path = os.path.join(image_dir, file_name)
        img = Image.open(img_path).convert("RGB")  # All preprocessed images are RGB
        tensor = transform(img)  # (3, H, W)
        pixel_sum += tensor.sum(dim=(1, 2))  # Sum over H and W
        pixel_squared_sum += (tensor**2).sum(dim=(1, 2))
        num_pixels += tensor.shape[1] * tensor.shape[2]

    mean = pixel_sum / num_pixels
    std = (pixel_squared_sum / num_pixels - mean**2).sqrt()
    return mean.tolist(), std.tolist()


if __name__ == "__main__":
    print(compute_mean_std(image_dir="./data/preprocessed/train"))
