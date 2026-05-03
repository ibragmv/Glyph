from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ImperialAramaicResNet18(nn.Module):
    def __init__(
        self,
        num_classes: int = 22,
        dropout: float = 0.3,
        pretrained: bool = True,
        small_image_stem: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)

        original_conv = model.conv1
        if small_image_stem:
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    averaged = original_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight.copy_(averaged[:, :, 2:5, 2:5])
            model.conv1 = new_conv
            model.maxpool = nn.Identity()
        else:
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    new_conv.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
            model.conv1 = new_conv

        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @property
    def layer4(self):
        return self.model.layer4


def build_model(
    num_classes: int = 22,
    dropout: float = 0.3,
    pretrained: bool = True,
    small_image_stem: bool = True,
) -> ImperialAramaicResNet18:
    return ImperialAramaicResNet18(
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained,
        small_image_stem=small_image_stem,
    )

