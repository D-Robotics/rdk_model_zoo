# Copyright (c) 2021 Bubbliiiing
# Copyright (c) 2026 D-Robotics Corporation
#
# Portions of this file are derived from bubbliiiing/unet-pytorch under the
# MIT License. See ../../../README.md#license for attribution.

"""Feature-only ResNet encoders used by the UNet model family."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Type

from torch import Tensor, nn


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    """Return a 3x3 convolution used by a ResNet residual block."""

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """Return a 1x1 projection convolution."""

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    """Standard two-convolution residual block used by ResNet18 and 34."""

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("BasicBlock does not support dilation > 1")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs: Tensor) -> Tensor:
        identity = inputs
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        if self.downsample is not None:
            identity = self.downsample(inputs)
        return self.relu(outputs + identity)


class Bottleneck(nn.Module):
    """Standard three-convolution block used by ResNet50, 101, and 152."""

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs: Tensor) -> Tensor:
        identity = inputs
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.relu(self.bn2(self.conv2(outputs)))
        outputs = self.bn3(self.conv3(outputs))
        if self.downsample is not None:
            identity = self.downsample(inputs)
        return self.relu(outputs + identity)


@dataclass(frozen=True)
class ResNetSpec:
    """Immutable structure and feature-channel contract for one encoder."""

    block: Type[nn.Module]
    layers: tuple[int, int, int, int]
    feature_channels: tuple[int, int, int, int, int]


RESNET_SPECS = {
    "resnet18": ResNetSpec(BasicBlock, (2, 2, 2, 2), (64, 64, 128, 256, 512)),
    "resnet34": ResNetSpec(BasicBlock, (3, 4, 6, 3), (64, 64, 128, 256, 512)),
    "resnet50": ResNetSpec(
        Bottleneck,
        (3, 4, 6, 3),
        (64, 256, 512, 1024, 2048),
    ),
    "resnet101": ResNetSpec(
        Bottleneck,
        (3, 4, 23, 3),
        (64, 256, 512, 1024, 2048),
    ),
    "resnet152": ResNetSpec(
        Bottleneck,
        (3, 8, 36, 3),
        (64, 256, 512, 1024, 2048),
    ),
}


class ResNetEncoder(nn.Module):
    """ResNet stem and stages returning five skip-connection features.

    The module names and ResNet50 pooling behavior intentionally match the
    upstream PR #128 model so that its published VOC checkpoint loads strictly.
    """

    def __init__(self, spec: ResNetSpec) -> None:
        super().__init__()
        self.inplanes = 64
        self.feature_channels = spec.feature_channels
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=0,
            ceil_mode=True,
        )
        self.layer1 = self._make_layer(spec.block, 64, spec.layers[0])
        self.layer2 = self._make_layer(spec.block, 128, spec.layers[1], stride=2)
        self.layer3 = self._make_layer(spec.block, 256, spec.layers[2], stride=2)
        self.layer4 = self._make_layer(spec.block, 512, spec.layers[3], stride=2)
        self._initialize_weights()

    def _make_layer(
        self,
        block: Type[nn.Module],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        expansion = int(block.expansion)
        downsample = None
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * expansion, stride),
                nn.BatchNorm2d(planes * expansion),
            )

        layers: list[nn.Module] = [
            block(self.inplanes, planes, stride=stride, downsample=downsample)
        ]
        self.inplanes = planes * expansion
        layers.extend(block(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_out = (
                    module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                )
                module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def load_torchvision_weights(self, backbone: str) -> None:
        """Load the default torchvision ImageNet weights into this encoder."""

        try:
            from torchvision.models import get_model, get_model_weights
        except ImportError as exc:
            raise RuntimeError(
                "torchvision is required to load ImageNet encoder weights"
            ) from exc

        weights = get_model_weights(backbone).DEFAULT
        reference = get_model(backbone, weights=weights)
        state_dict = {
            name: value
            for name, value in reference.state_dict().items()
            if not name.startswith("fc.")
        }
        result = self.load_state_dict(state_dict, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                "ImageNet encoder state mismatch: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )

    def forward(self, inputs: Tensor) -> List[Tensor]:
        feat1 = self.relu(self.bn1(self.conv1(inputs)))
        feat2 = self.layer1(self.maxpool(feat1))
        feat3 = self.layer2(feat2)
        feat4 = self.layer3(feat3)
        feat5 = self.layer4(feat4)
        return [feat1, feat2, feat3, feat4, feat5]


def build_resnet_encoder(
    backbone: str,
    pretrained: bool = False,
) -> ResNetEncoder:
    """Build one of the five supported feature-only ResNet encoders."""

    try:
        spec = RESNET_SPECS[backbone]
    except KeyError as exc:
        supported = ", ".join(RESNET_SPECS)
        raise ValueError(
            f"Unsupported backbone {backbone!r}; expected one of: {supported}"
        ) from exc

    encoder = ResNetEncoder(spec)
    if pretrained:
        encoder.load_torchvision_weights(backbone)
    return encoder


def feature_channels(backbone: str) -> Sequence[int]:
    """Return the five ordered feature-channel counts for a backbone."""

    try:
        return RESNET_SPECS[backbone].feature_channels
    except KeyError as exc:
        supported = ", ".join(RESNET_SPECS)
        raise ValueError(
            f"Unsupported backbone {backbone!r}; expected one of: {supported}"
        ) from exc
