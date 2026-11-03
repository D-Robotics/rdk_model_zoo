# TorchVision GoogLeNet

The independent crop224 profile uses [TorchVision v0.21.0](https://github.com/pytorch/vision/tree/7af698794eded568735f9519593603c1ec889eba),
commit `7af698794eded568735f9519593603c1ec889eba`, and its original
[BSD-3-Clause notice](licenses/LICENSE.torchvision).

Official checkpoint: <https://download.pytorch.org/models/googlenet-1378be20.pth>.
The observed complete SHA-256 is
`1378be20a8e875cf1568b8a71654e704449655e34711a959a38b04fb34905cef`;
the publisher supplies the `1378be20` digest prefix, not a complete SHA-256.

Strictly load all checkpoint keys, preserve `transform_input=True`, and remove
the unused auxiliary heads only after loading. The native deployment uses
Pillow bicubic short256 / rounded crop224 with ImageNet normalization and
float32 1,000-class logits. Its geometry differs from the upstream evaluation
preset; no upstream ImageNet accuracy is claimed for it. Historical default
artifacts and their measurements keep their original identities.
