[English](./README.md) | [简体中文](./README_cn.md)

# UNet Model Description

This sample provides a Pascal VOC semantic-segmentation deployment pipeline for
UNet with ResNet18, ResNet34, ResNet50, ResNet101, and ResNet152 backbones. It
covers checkpoint export, X5 PTQ conversion, accuracy evaluation, and Python
inference on RDK X5.

## Algorithm Overview

UNet uses an encoder-decoder structure with skip connections to combine
high-level semantics and fine spatial details. This implementation uses a
ResNet encoder and a UNet decoder to produce one class score for every pixel.

- Task: Pascal VOC semantic segmentation, 21 classes including background
- UNet paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- ResNet paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Reference implementation: [bubbliiiing/unet-pytorch](https://github.com/bubbliiiing/unet-pytorch)

### Deployment Contract

| Item | Contract |
| --- | --- |
| Target | RDK X5, `bayes-e` |
| Training input | RGB float32 NCHW `[1, 3, 512, 512]`, scaled by `1/255` |
| Runtime input | Packed NV12, 512 × 512 |
| Output | Float32 NCHW logits `[1, 21, 512, 512]` |
| Post-processing | `argmax` over the class dimension |

### Supported Backbones

| Backbone | Current status |
| --- | --- |
| ResNet18 | Prebuilt X5 BIN published; download and Python inference verified on RDK X5 |
| ResNet34 | FP32 mIoU 0.689319; ONNX export and X5 PTQ passed; BIN published, board Runtime pending |
| ResNet50 | FP32 mIoU 0.683826; ONNX export and X5 PTQ passed; BIN published, board Runtime pending |
| ResNet101 | FP32 mIoU 0.709437; ONNX export and X5 PTQ passed; BIN published, board Runtime pending |
| ResNet152 | FP32 mIoU 0.740002; ONNX export and X5 PTQ passed; BIN published, board Runtime pending |

The upstream ResNet50 VOC checkpoint is available from the
[`unet-pytorch` v1.0 release](https://github.com/bubbliiiing/unet-pytorch/releases/download/v1.0/unet_resnet_voc.pth),
SHA256 `556a74b8379c40cbc76af7a1faab84d1316f02b7d93290b5f1f724ff922faacb`.
Generated variants use torchvision ImageNet encoder initialization; training
receipts must record the exact torchvision version and weight identifier.

Sharing architecture code and conversion templates does not make an untested
backbone supported. Each variant must independently pass checkpoint, ONNX, PTQ,
accuracy, Runtime, and board-performance gates.

## Directory Structure

```text
unet/
├── conversion/                         # Checkpoint-to-X5 conversion
│   ├── mapper.py                       # Guarded checker and makertbin entry
│   ├── onnx_export/
│   │   ├── export_unet.py              # Strict checkpoint and ONNX exporter
│   │   └── model/                      # Shared UNet ResNet architecture
│   ├── ptq_yamls/                      # One bayes-e template per backbone
│   ├── README.md
│   └── README_cn.md
├── evaluator/                          # Unified PyTorch/ONNX/X5 accuracy entry
│   ├── eval_unet.py
│   ├── README.md
│   └── README_cn.md
├── model/                              # Prebuilt X5 models and downloads
│   ├── download_model.sh               # Download models by backbone
│   ├── README.md
│   └── README_cn.md
├── runtime/
│   └── python/                         # RDK X5 hbm_runtime sample
│       ├── unet.py                     # UNetConfig and UNet model wrapper
│       ├── main.py                     # Command-line inference entry
│       ├── run.sh                      # One-command launcher
│       ├── README.md
│       └── README_cn.md
├── test_data/                          # Default Pascal VOC test image
│   ├── 2007_000033.jpg
│   ├── README.md
│   └── README_cn.md
├── README.md
└── README_cn.md
```

Training utilities and generated intermediate artifacts are maintained outside
this sample. The repository does not commit checkpoints, ONNX files,
calibration data, compiled BIN files, or evaluation datasets; prebuilt BINs are
downloaded with the script in `model/`.

## QuickStart

The Python Runtime is the user-facing inference entry. Run with zero arguments;
the launcher downloads the default ResNet18 model when it is missing:

```bash
cd samples/vision/unet/runtime/python
./run.sh
```

The command loads the X5 BIN, converts the BGR image to packed NV12, runs BPU
inference, and writes a class-index mask, a colored overlay, and a JSON report.
See the [Python Runtime guide](./runtime/python/README.md) for parameters and API
details.

## Model Conversion

Ordinary users can skip conversion because precompiled models for all five
backbones are available.
Developers reproducing checkpoint export, calibration, checker, and makertbin
should follow the [conversion guide](./conversion/README.md).

## Runtime

The sample currently provides a Python implementation based on `hbm_runtime`.
See [runtime/python/README.md](./runtime/python/README.md) for environment setup,
default paths, CLI parameters, output files, and reusable interfaces.

## Evaluation

The unified evaluator measures the same Pascal VOC manifest with a PyTorch
checkpoint, an ONNX model, or an X5 BIN. See
[evaluator/README.md](./evaluator/README.md).

## Reference Results

The following results were measured on all 1,449 Pascal VOC validation samples
using the earlier ResNet18 checkpoint that validated the deployment pipeline.
They are not a re-evaluation of the current download and do not represent the
other backbones. A maintainer subsequently verified that the published ResNet18
model downloads and produces its mask and overlay on RDK X5; board accuracy and
pure BPU performance revalidation remain pending.

| Backend | mIoU | Pixel Accuracy |
| --- | ---: | ---: |
| PyTorch FP32 | 0.619695 | 0.911532 |
| ONNX Runtime FP32 | 0.619694 | 0.911532 |
| RDK X5 PTQ | 0.617198 | 0.910332 |

On RDK X5, `hrt_model_exec` measured 52.72 ms average latency and 18.96 FPS
with one thread, 200 frames, and a real packed NV12 input. End-to-end Python
latency also includes image decoding, preprocessing, output transfer, and
post-processing.

### ResNet34/50/101/152 Release Results

The four newly trained variants were evaluated on the same complete 1,449-image
validation set. Their ONNX numerical gates and `bayes-e` PTQ compilation passed,
and their public BIN files were downloaded again and checked against the
published SHA256 values. Board Runtime accuracy and performance were not run for
these four releases.

| Backbone | PyTorch FP32 mIoU | Pixel Accuracy | PTQ Output Cosine | Board Runtime |
| --- | ---: | ---: | ---: | --- |
| ResNet34 | 0.689319 | 0.930947 | 0.998328 | Pending |
| ResNet50 | 0.683826 | 0.928404 | 0.995292 | Pending |
| ResNet101 | 0.709437 | 0.935887 | 0.996384 | Pending |
| ResNet152 | 0.740002 | 0.942805 | 0.996070 | Pending |

ResNet50 improved over the upstream checkpoint's 0.661483 mIoU but did not
exceed the ResNet34 result. Download URLs and SHA256 values for the complete
family are listed in the [model guide](./model/README.md).

## License

This sample follows the repository-level [Apache License 2.0](../../../LICENSE).
The reference UNet implementation is derived from
[`bubbliiiing/unet-pytorch`](https://github.com/bubbliiiing/unet-pytorch) at
commit `5bcd6b4c832648beed1b92e78ed1e85c56343eca` and retains its MIT terms.

<details>
<summary>MIT notice for derived UNet code</summary>

```text
MIT License

Copyright (c) 2021 Bubbliiiing

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

</details>
