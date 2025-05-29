import copy
import torch
from torchvision import transforms  # type:ignore
from PIL import Image

from utils.label_text_mapping import labels_to_text
from utils.resize_and_pad import ResizeAndPadTransform, resize_and_pad
from .ocr_model import OCRModel
from config.model_config import OCRModelConfig


class OCRModelWrapper:
    def __init__(self, model: OCRModel, config: OCRModelConfig, device: torch.device):
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device

        if device.type == "mps":
            print("Using MPS with compatibility mode")
            self.cpu_model = model
            self.model = copy.deepcopy(model).to(device)
            self.model.eval()
        else:
            self.model = model.to(device)
            self.model.eval()

        self.transform = transforms.Compose(
            [
                # transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),
                # transforms.Grayscale(num_output_channels=1),
                ResizeAndPadTransform(self.config.height, self.config.width),
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

    def predict(self, image_input: str | Image.Image | torch.Tensor):
        with torch.no_grad():
            image_tensor = self.preprocess_image(image_input).to(self.device)
            batch_size = image_tensor.size(0)
            memory = self.model.encode(image_tensor)  # Precomputed encoder output

            # Start with SOS token
            trg_tensor = torch.full(
                (1, batch_size), self.config.char2idx["SOS"], device=self.device
            )
            active = torch.ones(batch_size, dtype=torch.bool, device=self.device)

            for _ in range(self.config.max_length):
                output = self.model.decode(trg_tensor, memory)  # Decode all sequences
                out_tokens = output.argmax(2)[
                    -1
                ]  # Predict next token for each sequence

                # Append new tokens, padding inactive sequences
                new_tokens = torch.where(
                    active, out_tokens, self.config.char2idx["PAD"]
                )
                trg_tensor = torch.cat([trg_tensor, new_tokens.unsqueeze(0)], dim=0)

                # Update which sequences are still active
                active = active & (out_tokens != self.config.char2idx["EOS"])
                if not active.any():  # Stop if all sequences have EOS
                    break

            # Convert sequences to text
            predicted_texts = []
            for b in range(batch_size):
                seq = trg_tensor[1:, b]
                eos_idx = (seq == self.config.char2idx["EOS"]).nonzero()
                if len(eos_idx) > 0:
                    seq = seq[: eos_idx[0]]
                predicted_texts.append(labels_to_text(seq.tolist(), self.config))

            return predicted_texts
