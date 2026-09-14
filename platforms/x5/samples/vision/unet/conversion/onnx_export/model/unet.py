# Copyright (c) 2021 Bubbliiiing
# Copyright (c) 2026 D-Robotics Corporation
#
# Portions of this file are derived from bubbliiiing/unet-pytorch under the
# MIT License. See ../../../README.md#license for attribution.

"""UNet decoder shared by the five supported ResNet encoders."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn

from .resnet import RESNET_SPECS, build_resnet_encoder, feature_channels


class unetUp(nn.Module):
    """Upsample, concatenate one encoder skip, and refine the feature map.

    The historical class and member names are retained for strict compatibility
    with the published ResNet50 VOC checkpoint.
    """

    def __init__(self, in_size: int, out_size: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, skip: Tensor, decoder: Tensor) -> Tensor:
        outputs = torch.cat([skip, self.up(decoder)], dim=1)
        outputs = self.relu(self.conv1(outputs))
        return self.relu(self.conv2(outputs))


class UNet(nn.Module):
    """UNet semantic-segmentation model with a configurable ResNet encoder."""

    def __init__(
        self,
        num_classes: int = 21,
        pretrained: bool = False,
        backbone: str = "resnet50",
    ) -> None:
        super().__init__()
        if backbone not in RESNET_SPECS:
            supported = ", ".join(RESNET_SPECS)
            raise ValueError(
                f"Unsupported backbone {backbone!r}; expected one of: {supported}"
            )

        self.resnet = build_resnet_encoder(backbone, pretrained=pretrained)
        channels = feature_channels(backbone)
        out_filters = [64, 128, 256, 512]
        in_filters = [
            channels[0] + out_filters[1],
            channels[1] + out_filters[2],
            channels[2] + out_filters[3],
            channels[3] + channels[4],
        ]

        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])
        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.final = nn.Conv2d(out_filters[0], num_classes, kernel_size=1)
        self.backbone = backbone
        self.num_classes = num_classes

    @property
    def deployment_contract(self) -> Dict[str, object]:
        """Return model metadata used by export and validation tooling."""

        return {
            "backbone": self.backbone,
            "num_classes": self.num_classes,
            "input_shape": [1, 3, 512, 512],
            "output_shape": [1, self.num_classes, 512, 512],
        }

    def forward(self, inputs: Tensor) -> Tensor:
        feat1, feat2, feat3, feat4, feat5 = self.resnet(inputs)
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)
        return self.final(self.up_conv(up1))

    def freeze_backbone(self) -> None:
        """Disable gradients for all encoder parameters."""

        for parameter in self.resnet.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Enable gradients for all encoder parameters."""

        for parameter in self.resnet.parameters():
            parameter.requires_grad = True


# Preserve the original public spelling used by the upstream checkpoint code.
Unet = UNet
