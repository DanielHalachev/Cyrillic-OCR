import torch
import torch.nn as nn
from torchvision import models  # type: ignore
from config.resnet_type import ResNetType, ResNetWeights


class CNNBackbone(nn.Module):
    """
    CNN Backbone for feature extraction.
    """

    def __init__(
        self,
        hidden: int,
        pretrained: bool = False,
        backbone_type: ResNetType = ResNetType.RESNET50,
    ):
        """
        Initialize the CNN backbone.

        Args:
            hidden (int): Output feature dimension for the transformer.
            pretrained (bool): Whether to use pre-trained weights.
            backbone_type (ResNetType): Type of ResNet backbone.
        """
        super(CNNBackbone, self).__init__()

        supported_backbones = [e.value for e in ResNetType]
        if backbone_type not in supported_backbones:
            raise ValueError(
                f"Unsupported backbone_type: {backbone_type.value}. Choose from {supported_backbones}"
            )

        if hidden % 2 != 0 or hidden <= 0:
            raise ValueError(f"hidden must be a positive even number, got {hidden}")

        weights = (
            getattr(models, ResNetWeights.weights[backbone_type]).DEFAULT
            if pretrained
            else None
        )
        self.backbone = getattr(models, backbone_type.value)(weights=weights)
        in_channels = 512 if backbone_type in ["resnet18", "resnet34"] else 2048

        if pretrained:
            conv1_weight = self.backbone.conv1.weight  # Shape: [64, 3, 7, 7]
            # Average across the 3 input channels
            conv1_weight_1ch = conv1_weight.mean(
                dim=1, keepdim=True
            )  # Shape: [64, 1, 7, 7]
            self.backbone.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.backbone.conv1.weight = nn.Parameter(conv1_weight_1ch)
        else:
            self.backbone.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Replace fc with a 1x1 convolution to output hidden channels
        self.backbone.fc = nn.Conv2d(in_channels, hidden, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNet.

        Args:
            x (Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            Tensor: Output tensor of shape [W, B, hidden].
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)  # [B, in_channels, H, W]

        x = nn.AdaptiveAvgPool2d((1, None))(x)  # [B, in_channels, 1, W]
        x = self.backbone.fc(x)  # [B, hidden, 1, W]
        x = x.squeeze(2)  # [B, hidden, W]
        x = x.permute(2, 0, 1)  # [W, B, hidden]
        return x
