from argparse import ArgumentParser
from PIL import Image
import os
from pathlib import Path
from utils.resize_and_pad import resize_and_pad
from tqdm import tqdm


def preprocess_images(image_dir, output_dir, height=64, width=256):
    os.makedirs(output_dir, exist_ok=True)
    for img_name in tqdm(os.listdir(image_dir)):
        img_path = os.path.join(image_dir, img_name)
        img = Image.open(img_path).convert("L")  # Grayscale
        img = resize_and_pad(img, height, width)
        output_path = os.path.join(output_dir, img_name)
        img.save(output_path, "PNG")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=str,
        default="data/",
        help="Directory containing training images",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="data/processed",
        help="Directory containing training images",
    )

    args = parser.parse_args()
    preprocess_images(
        os.path.join(args.source_dir, "train"), os.path.join(args.target_dir, "train")
    )
    preprocess_images(
        os.path.join(args.source_dir, "test"), os.path.join(args.target_dir, "test")
    )
