from argparse import ArgumentParser
from PIL import Image
import os
from utils.resize_and_pad import ResizeAndPadTransform
from tqdm import tqdm
from torchvision import transforms  # type:ignore


def preprocess_images(image_dir, output_dir, height=64, width=256):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            ResizeAndPadTransform(height=64, width=256),
            # transforms.RandomRotation(degrees=(-9, 9), fill=255),
            # transforms.RandomAffine(degrees=5, scale=(0.9, 1.1), shear=2),
            transforms.ToTensor(),
            transforms.ToPILImage(),
        ]
    )
    os.makedirs(output_dir, exist_ok=True)
    for img_name in tqdm(os.listdir(image_dir)):
        img_path = os.path.join(image_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img = transform(img)
        output_path = os.path.join(output_dir, img_name)
        img.save(output_path, "PNG")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=str,
        default="data/raw",
        help="Directory containing training images",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="data/preprocessed",
        help="Directory containing training images",
    )

    args = parser.parse_args()
    preprocess_images(
        os.path.join(args.source_dir, "train"), os.path.join(args.target_dir, "train")
    )
    preprocess_images(
        os.path.join(args.source_dir, "test"), os.path.join(args.target_dir, "test")
    )
