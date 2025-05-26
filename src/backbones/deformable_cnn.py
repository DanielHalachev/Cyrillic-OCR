import torch
import torch.nn as nn
from torchvision import models  # type: ignore
from torchvision.ops import DeformConv2d  # type: ignore
from config.resnet_type import ResNetType, ResNetWeights


class DeformableBottleneck(nn.Module):
    def __init__(
        self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d
    ):
        super(DeformableBottleneck, self).__init__()

        width = planes
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        nn.init.constant_(self.bn1.weight, 1.0)
        nn.init.constant_(self.bn1.bias, 0.0)

        self.offset_conv = nn.Conv2d(
            width, 2 * 3 * 3, kernel_size=3, stride=stride, padding=1, bias=True
        )
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)  # type:ignore

        self.deform_conv = DeformConv2d(
            width, width, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = norm_layer(width)
        nn.init.constant_(self.bn2.weight, 1.0)
        nn.init.constant_(self.bn2.bias, 0.0)

        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        nn.init.constant_(self.bn3.weight, 1.0)
        nn.init.constant_(self.bn3.bias, 0.0)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        offset = self.offset_conv(out)
        out = self.deform_conv(out, offset)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DeformableResNetBackbone(nn.Module):
    def __init__(
        self,
        backbone_type: ResNetType = ResNetType.RESNET50,
        pretrained: bool = True,
        hidden: int = 512,
    ):
        super(DeformableResNetBackbone, self).__init__()

        if hidden % 2 != 0 or hidden <= 0:
            raise ValueError(f"hidden must be a positive even number, got {hidden}")

        supported_backbones = [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
        ]
        if backbone_type.value not in supported_backbones:
            raise ValueError(
                f"Unsupported backbone_type: {backbone_type.value}. Choose from {supported_backbones}"
            )

        weights = (
            getattr(models, ResNetWeights.weights[backbone_type]).DEFAULT
            if pretrained
            else None
        )
        self.model = getattr(models, backbone_type.value)(weights=weights)
        self.norm_layer = nn.BatchNorm2d

        if backbone_type.value in ["resnet18", "resnet34"]:
            layer3_params = {"inplanes": 256, "planes": 128}
            layer4_params = {"inplanes": 512, "planes": 256}
            in_channels = 512
        else:
            layer3_params = {"inplanes": 512, "planes": 256}
            layer4_params = {"inplanes": 1024, "planes": 512}
            in_channels = 2048

        self.model.layer3 = self._replace_with_dcn(
            **layer3_params, num_blocks=len(self.model.layer3), stride=2
        )
        self.model.layer4 = self._replace_with_dcn(
            **layer4_params, num_blocks=len(self.model.layer4), stride=2
        )

        self.reduction_conv = nn.Conv2d(in_channels, hidden, kernel_size=1)

    def _replace_with_dcn(self, inplanes, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes, planes * 4, kernel_size=1, stride=stride, bias=False
                ),
                self.norm_layer(planes * 4),
            )

        layers = [
            DeformableBottleneck(inplanes, planes, stride, downsample, self.norm_layer)
        ]
        inplanes = planes * 4
        for _ in range(1, num_blocks):
            layers.append(
                DeformableBottleneck(inplanes, planes, norm_layer=self.norm_layer)
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)  # [B, in_channels, H, W]

        x = nn.AdaptiveAvgPool2d((1, None))(x)  # [B, in_channels, 1, W]
        x = self.reduction_conv(x)  # [B, hidden, 1, W]
        x = x.squeeze(2)  # [B, hidden, W]
        x = x.permute(2, 0, 1)  # [W, B, hidden]

        return x
