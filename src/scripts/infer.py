import argparse
import os
from pathlib import Path
import numpy as np
import torch
import traceback
import cv2
import matplotlib.image as mpimg
import easyocr  # type:ignore
from torchvision import transforms  # type:ignore
from config.model_config import OCRModelConfig
from models.ocr_model_wrapper import OCRModelWrapper
from models.ocr_model import ResnetOCRModel


class InferenceModule:
    def image_region_inference(self, model_wrapper: OCRModelWrapper, image_input):
        return model_wrapper.predict(image_input)

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


def main(args):
    reader = easyocr.Reader(["bg", "ru"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = OCRModelConfig()
    model = ResnetOCRModel(config)
    model.load_from_checkpoint(
        Path(os.path.expanduser("~/synthentic-natural-resnet-50.pt")), device, *config
    )
    model_wrapper = OCRModelWrapper(model, config, device)

    inferenceModule = InferenceModule()

    # Perform inference on each image in the input directory
    if args.image_file:
        image_files = [os.path.basename(args.image_file)]
    else:
        image_files = os.listdir(args.input_dir)

    for image_path in image_files:
        # Check if the file is an image
        if image_path.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                print(f"Processing image: {image_path}")

                img = mpimg.imread(image_path)

                detected_bboxes = reader.readtext(image_path)
                detected_bboxes.sort(
                    key=lambda bbox: (
                        np.mean([pt[1] for pt in bbox[0]]),  # type: ignore
                        np.mean([pt[0] for pt in bbox[0]]),  # type: ignore
                    )
                )

                results = []
                for i, detected_bbox in enumerate(detected_bboxes):
                    bbox, label, score = detected_bbox
                    bbox = [[int(coordinate) for coordinate in point] for point in bbox]
                    cropped_img = img[bbox[0][1] : bbox[2][1], bbox[0][0] : bbox[2][0]]
                    if cropped_img.size > 0:
                        predicted_transcript = inferenceModule.image_region_inference(
                            model_wrapper, cropped_img
                        )
                        results.append([str(bbox), score, predicted_transcript])

                return results
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/images/",
        help="Directory containing input images",
    )
    parser.add_argument(
        "--image-file", type=str, default=None, help="Single image file to process"
    )
    args = parser.parse_args()
    main(args)
