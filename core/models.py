from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
    resnet18,
    resnet34,
)


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    builder: Callable[..., nn.Module]
    weights: Any
    family: str


BACKBONE_SPECS: dict[str, BackboneSpec] = {
    "resnet18": BackboneSpec(
        name="resnet18",
        builder=resnet18,
        weights=ResNet18_Weights,
        family="resnet",
    ),
    "resnet34": BackboneSpec(
        name="resnet34",
        builder=resnet34,
        weights=ResNet34_Weights,
        family="resnet",
    ),
    "efficientnet_b0": BackboneSpec(
        name="efficientnet_b0",
        builder=efficientnet_b0,
        weights=EfficientNet_B0_Weights,
        family="efficientnet",
    ),
    "mobilenet_v3_small": BackboneSpec(
        name="mobilenet_v3_small",
        builder=mobilenet_v3_small,
        weights=MobileNet_V3_Small_Weights,
        family="mobilenet_v3",
    ),
}

DEFAULT_BACKBONE = "resnet18"


def get_available_backbones() -> tuple[str, ...]:
    return tuple(BACKBONE_SPECS)


def get_backbone_spec(backbone_name: str) -> BackboneSpec:
    if backbone_name not in BACKBONE_SPECS:
        available = ", ".join(BACKBONE_SPECS)
        raise ValueError(
            f"Unsupported backbone {backbone_name!r}. Available: {available}"
        )
    return BACKBONE_SPECS[backbone_name]


def _find_last_conv_layer(module: nn.Module) -> nn.Module | None:
    for child in reversed(list(module.children())):
        target = _find_last_conv_layer(child)
        if target is not None:
            return target
    if isinstance(module, nn.Conv2d):
        return module
    return None


def _resize_stem_conv(
    conv: nn.Conv2d,
    *,
    kernel_size: int | None = None,
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] | None = None,
) -> nn.Conv2d:
    kernel = kernel_size if kernel_size is not None else conv.kernel_size[0]
    new_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=kernel,
        stride=stride if stride is not None else conv.stride,
        padding=padding if padding is not None else conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        if kernel == conv.kernel_size[0]:
            new_conv.weight.copy_(conv.weight)
        else:
            center = conv.kernel_size[0] // 2
            radius = kernel // 2
            new_conv.weight.copy_(
                conv.weight[
                    :,
                    :,
                    center - radius : center + radius + 1,
                    center - radius : center + radius + 1,
                ]
            )
        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


class ImperialAramaicClassifier(nn.Module):
    def __init__(
        self,
        *,
        backbone_name: str = DEFAULT_BACKBONE,
        num_classes: int = 22,
        dropout: float = 0.3,
        pretrained: bool = True,
        small_image_stem: bool = True,
    ) -> None:
        super().__init__()
        spec = get_backbone_spec(backbone_name)
        weights = spec.weights.DEFAULT if pretrained else None
        model = spec.builder(weights=weights)

        if spec.family == "resnet":
            original_conv = model.conv1
            if small_image_stem:
                model.conv1 = _resize_stem_conv(
                    original_conv,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
                model.maxpool = nn.Identity()
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
            feature_root = model.layer4
        elif spec.family == "efficientnet":
            in_features = model.classifier[-1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
            feature_root = model.features
        elif spec.family == "mobilenet_v3":
            in_features = model.classifier[-1].in_features
            model.classifier = nn.Sequential(
                model.classifier[0],
                model.classifier[1],
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(in_features, num_classes),
            )
            feature_root = model.features
        else:
            raise ValueError(f"Unsupported backbone family: {spec.family}")

        self.model = model
        self.backbone_name = spec.name
        self.backbone_family = spec.family
        self.small_image_stem = bool(small_image_stem and spec.family == "resnet")
        gradcam_layer = _find_last_conv_layer(feature_root)
        if gradcam_layer is None:
            raise RuntimeError(f"Could not determine Grad-CAM layer for {spec.name}")
        self.gradcam_layer = gradcam_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_model(
    num_classes: int = 22,
    dropout: float = 0.3,
    pretrained: bool = True,
    small_image_stem: bool = True,
    backbone_name: str = DEFAULT_BACKBONE,
) -> ImperialAramaicClassifier:
    return ImperialAramaicClassifier(
        backbone_name=backbone_name,
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained,
        small_image_stem=small_image_stem,
    )
