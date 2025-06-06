import os, sys
from pathlib import Path
import tempfile
from flask import Flask, render_template, request, send_from_directory, url_for
import torch
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract  # type:ignore
from torchvision import transforms  # type:ignore

# Allow Python to reference modules, defined in the parent directory
d = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, d)

from backbones.cnn import CNNBackbone
from config.model_config import OCRModelConfig
from models.ocr_model import OCRModel
from src.models.ocr_model_wrapper import OCRModelWrapper

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = tempfile.mkdtemp()
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

# Load device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Load model once
cfg = OCRModelConfig()
model = OCRModel.load_from_checkpoint(
    model_path=Path("model.pt"),
    device=device,
    backbone=CNNBackbone(cfg.hidden, pretrained=True),
    outtoken=len(cfg.tokens),
    hidden=cfg.hidden,
    enc_layers=cfg.enc_layers,
    dec_layers=cfg.dec_layers,
    nhead=cfg.nhead,
    dropout=cfg.dropout,
)
model.eval()
wrapper = OCRModelWrapper(model, cfg, device)


def extract_words(image: Image.Image):
    data = pytesseract.image_to_data(
        image, lang="ru", output_type=pytesseract.Output.DICT
    )

    boxes = []
    for i in range(len(data["text"])):
        if data["text"][i].strip():  # Ignore empty or whitespace-only text
            box = (
                data["left"][i],
                data["top"][i],
                data["left"][i] + data["width"][i],
                data["top"][i] + data["height"][i],
            )
            boxes.append(box)

    boxes.sort(key=lambda b: (b[1], b[0]))

    word_images = [image.crop(box) for box in boxes]

    # return word_images

    transform = transforms.ToTensor()
    word_tensors = [transform(img) for img in word_images]
    return word_tensors


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    result_text = None
    filename = request.form.get("filename")  # Preserve filename across requests

    if request.method == "POST":
        f = request.files.get("image")
        extra = "extra_processing" in request.form

        if not f or f.filename == "":
            error = "No file selected"
        else:
            # Save upload
            filename = secure_filename(f.filename)  # type:ignore
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            f.save(save_path)

            try:
                img = Image.open(save_path).convert("RGB")
                if extra:
                    image_tensors = extract_words(img)
                    result_text = " ".join(
                        [
                            " ".join(wrapper.predict(cropped_img_tensor))
                            for cropped_img_tensor in image_tensors
                        ]
                    )
                else:
                    result_text = "".join(wrapper.predict(img))
            except Exception as e:
                error = f"Error processing image: {e}"

    return render_template(
        "index.html", error=error, text=result_text, filename=filename
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True, port=8000)
