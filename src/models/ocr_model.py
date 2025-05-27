import os
import torch
import torch.nn as nn
import math

from backbones.cnn import CNNBackbone
from backbones.deformable_cnn import DeformableResNetBackbone
from config.model_config import OCRModelConfig


class OCRModel(nn.Module):

    def __init__(
        self,
        backbone: CNNBackbone | DeformableResNetBackbone,
        outtoken,
        hidden,
        enc_layers,
        dec_layers,
        nhead,
        dropout,
    ):
        super(OCRModel, self).__init__()
        self.backbone = backbone

        self.pos_encoder = PositionalEncoding(hidden, dropout, batch_first=True)
        self.decoder = nn.Embedding(outtoken, hidden)
        self.pos_decoder = PositionalEncoding(hidden, dropout, batch_first=True)
        self.transformer = nn.Transformer(
            d_model=hidden,
            nhead=nhead,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )

        self.fc_out = nn.Linear(hidden, outtoken)
        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def make_len_mask(self, inp):
        mask = (inp == 0).transpose(0, 1)
        return mask.float().masked_fill(mask, float("-inf"))

    def forward(self, src, trg):
        # src: [B, C, H, W]
        # trg: [T, B], token indices from dataloader
        x = self.backbone.forward(src)  # [B, W, hidden]
        src = self.pos_encoder(x)  # [B, W, hidden]
        src_pad_mask = (
            x[:, :, 0] == 0
        )  # [B, W], assuming first feature indicates padding

        # Compute target padding mask from token indices
        trg_input = trg.transpose(0, 1)  # [B, T], token indices
        trg_pad_mask = trg_input == 0  # [B, T]
        trg_embed = self.decoder(trg_input)  # [B, T, hidden]
        trg = self.pos_decoder(trg_embed)  # [B, T, hidden]

        output = self.transformer(
            src,
            trg,
            src_mask=self.src_mask,
            tgt_mask=self.trg_mask,
            memory_mask=self.memory_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=trg_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )  # [B, T, hidden]
        # ... rest of the forward pass
        return output

    def save_model(self, save_path: os.PathLike):
        """Save the full model to the specified path."""
        torch.save(self.state_dict(), save_path)

    def load_model(self, model_path: os.PathLike, device):
        """Load model weights from the specified path."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        state_dict = torch.load(model_path, device)
        self.load_state_dict(state_dict)

    @classmethod
    def load_from_checkpoint(cls, model_path: os.PathLike, device, **model_args):
        """Create a model instance and load weights from checkpoint."""
        model = cls(**model_args)
        model.load_model(model_path, device)
        return model

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        x = self.backbone(src)  # [B, W, hidden]
        src = self.pos_encoder(x)  # [B, W, hidden]
        memory = self.transformer.encoder(src)  # [B, W, hidden]
        return memory

    def decode(self, trg: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        trg = trg.transpose(0, 1)  # [B, T]
        trg_embed = self.decoder(trg)  # [B, T, hidden]
        trg_pos = self.pos_decoder(trg_embed)  # [B, T, hidden]
        output = self.transformer.decoder(trg_pos, memory)  # [B, T, hidden]
        output = self.fc_out(output)  # [B, T, vocab_size]
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))
        self.batch_first = batch_first
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = (
            pe.unsqueeze(0) if batch_first else pe.unsqueeze(1)
        )  # [1, max_len, d_model] or [max_len, 1, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        if self.batch_first:
            # [B, seq_len, d_model]
            x = x + self.scale * self.pe[:, : x.size(1), :]  # type:ignore
        else:
            # [seq_len, B, d_model]
            x = x + self.scale * self.pe[: x.size(0), :, :]  # type:ignore
        return self.dropout(x)


class ResnetOCRModel(OCRModel):
    def __init__(self, config: OCRModelConfig):
        backbone = CNNBackbone(config.hidden, pretrained=True)
        super().__init__(
            backbone,
            len(config.tokens),
            config.hidden,
            config.enc_layers,
            config.dec_layers,
            config.nhead,
            config.dropout,
        )


class DeformableResnetOCRModel(OCRModel):
    def __init__(self, backbone_path: os.PathLike, config: OCRModelConfig):
        backbone = DeformableResNetBackbone()
        if backbone_path and os.path.exists(backbone_path):
            state_dict = torch.load(backbone_path, map_location="cpu")
            # Check if the state dict belongs to the full model or just the backbone
            if "backbone" in state_dict:
                # It's a full model state dict
                backbone.load_state_dict(state_dict["backbone"])
            else:
                # It's just the backbone state dict
                backbone.load_state_dict(state_dict)
        super().__init__(
            backbone,
            len(config.tokens),
            config.hidden,
            config.enc_layers,
            config.dec_layers,
            config.nhead,
            config.dropout,
        )
