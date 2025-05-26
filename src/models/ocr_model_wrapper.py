import torch
from torchvision import transforms  # type:ignore
from PIL import Image

from utils.label_text_mapping import labels_to_text
from utils.resize_and_pad import resize_and_pad
from .ocr_model import OCRModel
from config.model_config import OCRModelConfig


class OCRModelWrapper:
    def __init__(self, model: OCRModel, config: OCRModelConfig, device: torch.device):
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Grayscale(num_output_channels=3),
                transforms.Grayscale(num_output_channels=1),
                transforms.Lambda(
                    lambda img: resize_and_pad(
                        img, self.config.height, self.config.width
                    )
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def preprocess_image(self, image_input: str | Image.Image | torch.Tensor):
        """Convert various image inputs to tensor"""
        if isinstance(image_input, str):
            # It's a path
            image = Image.open(image_input).convert("L")
            tensor = self.transform(image).unsqueeze(0)  # type:ignore
        elif isinstance(image_input, Image.Image):
            # It's a PIL image
            image = image_input.convert("L")
            tensor = self.transform(image).unsqueeze(0)  # type:ignore
        elif isinstance(image_input, torch.Tensor):
            # It's already a tensor - just ensure correct shape
            if image_input.dim() == 3:  # [C,H,W]
                tensor = image_input.unsqueeze(0)
            else:
                tensor = image_input  # Assume it's already [B,C,H,W]
        else:
            raise TypeError(f"Unsupported input type: {type(image_input)}")

        return tensor.to(self.device)

    def predict(self, image_input: str | Image.Image | torch.Tensor) -> list[str]:
        with torch.no_grad():
            image_tensor = self.preprocess_image(image_input)  # [B, C, H, W]
            batch_size = image_tensor.size(0)
            out_indexes_list = [
                torch.full(
                    (batch_size,), self.config.char2idx["SOS"], device=self.device
                )
            ]
            trg_tensor = (
                torch.LongTensor(out_indexes_list).transpose(0, 1).to(self.device)
            )  # [1, B]

            for _ in range(self.config.max_length):
                output = self.model(image_tensor, trg_tensor)  # [T, B, outtoken]
                out_tokens = output.argmax(2)[-1]  # [B]
                out_indexes_list.append(out_tokens)
                trg_tensor = torch.cat(
                    [trg_tensor, out_tokens.unsqueeze(0)], dim=0
                )  # [T+1, B]
                if (out_tokens == self.config.char2idx["EOS"]).all():
                    break

            out_indexes = torch.stack(out_indexes_list[1:], dim=0).transpose(
                0, 1
            )  # [B, T]
            predicted_texts = [
                labels_to_text(idxs.tolist(), self.config) for idxs in out_indexes
            ]
            return predicted_texts
