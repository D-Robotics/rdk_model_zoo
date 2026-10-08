# ConvNeXt image classification

ConvNeXt is a convolutional classifier built with large depthwise kernels and Transformer-style blocks.

Sources: [A ConvNet for the
2020s](https://arxiv.org/abs/2201.03545) · [facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)

[中文说明](README_cn.md)

<a id="overview"></a>

## Overview

ConvNeXt is a pure convolutional network modernized from the original
ResNet by progressively adopting designs borrowed from the Swin
Transformer ("A ConvNet for the 2020s"). It targets ImageNet-1k
1000-class image classification and outputs Top-K classes with
confidence scores. Four design changes distinguish it from a classic
ResNet:

- **Large-kernel depthwise convolution** — a 7×7 depthwise convolution
  replaces the traditional 3×3 convolutions, enlarging the receptive
  field at a MobileNet/EfficientNet-like parameter and compute cost.
- **Fewer activation functions, GELU instead of ReLU** — activation
  layers are sparser and the nonlinearity follows the Transformer style.
- **LayerNorm instead of BatchNorm** — more robust for small-batch data.
- **Simplified residual design** — the fully connected part is slimmed
  down and the ResNet bottleneck structure is removed.

![ConvNeXt block compared with the ResNet and Swin Transformer
blocks](./test_data/ConvNeXt_Block.png)

*Figure: block comparison from the ConvNeXt paper — Swin Transformer
block (left), ResNet block (middle), ConvNeXt block (right). The figure
shows the upstream training architecture; the artifact deployed on X5 is
the INT8-quantized atto variant at 224×224 NV12 (see
[Support matrix](#support-matrix)).*

Start with `runtime/python/main.py`; `classify.py` holds the model stages and `cli.py` handles options and results.
[runtime/python/README.md](runtime/python/README.md)

<a id="directory"></a>
## Directory structure

```text
convnext/
├── conversion/  # Export and quantization configuration
├── evaluator/  # Evaluation commands and metrics
├── model/  # Model files and download scripts
├── runtime/  # Python inference
├── test_data/  # Example inputs
├── tests/  # Automated tests
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
└── requirements-host.txt  # Python dependencies
```

<a id="support-matrix"></a>
## Support matrix

| Target | Variant | Language | Status |
| --- | --- | --- | --- |
| x5 | atto | python | supported |
| s100 / s100p / s600 | any | python | not-supported (the S manifest publishes no ConvNeXt asset; selection is an explicit error, no cross-platform fallback) |

<a id="prerequisites"></a>
## Prerequisites

Board Python execution needs the board image's matching `hbm_runtime`,
NumPy, and OpenCV-Python; the SDK is imported only when a model executes
(`--help`, `--list-models`, `--dry-run`, and host tests need no SDK).
Manifest reading also needs PyYAML. On a development host, install the
user-space dependencies from `requirements-host.txt`:

```bash
# cwd: repository root — success: "host dependencies: ok"
python3 -m venv .venv-convnext
source .venv-convnext/bin/activate
python3 -m pip install -r samples/vision/convnext/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## Quick start

One complete path on an X5 board, commands run from the repository root.
Prerequisite: the board image with `hbm_runtime` and network access to the
manifest's model server.

```bash
# 1. Prepare the artifact (input: manifest row x5:convnext:ConvNeXt_atto_224x224_nv12.bin)
#    output: samples/vision/convnext/model/ConvNeXt_atto_224x224_nv12.bin
#    success: downloader exits 0 and prints the observed digest
bash samples/vision/convnext/model/download.sh x5

# 2. Run classification (input: the artifact above plus the bundled test image)
#    output: Top-5 class ids, scores, labels on stdout
#    success: exit code 0 and a printed Top-5 list
python3 samples/vision/convnext/runtime/python/main.py \
  --target x5 \
  --asset-id x5:convnext:ConvNeXt_atto_224x224_nv12.bin \
  --model-path samples/vision/convnext/model/ConvNeXt_atto_224x224_nv12.bin \
  --test-img samples/vision/convnext/test_data/cheetah.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

The default variant is `atto`. Full commands:
[runtime/python/README.md](runtime/python/README.md).

<a id="expected-results"></a>
## Expected results

The Python run prints a stable Top-K (default 5) of class IDs, scores, and
labels and exits 0; Pass `--img-save-path` to save a visualization; otherwise results are printed to stdout. With the bundled `cheetah.JPEG` the Top-5 contains cheetah-related
ImageNet classes. Select the target from the support matrix and prepare its matching artifact before inference; the runtime checks board identity before loading the model.

The screenshot below shows a reference run from the X5 release: the demo
draws the top-1 label onto the bundled `cheetah.JPEG` (the visualization
written by `--img-save-path`); the recorded run returned class 293
`cheetah, chetah, Acinonyx jubatus` with score 0.8048811.

![Reference inference result on X5: the top-1 label drawn on the bundled
cheetah.JPEG, score 0.8048811](./test_data/inference.png)

<a id="performance"></a>
## Performance data

Published performance of the ConvNeXt series on RDK X5 (X5 release
x5-v1.1.3; Float Top-1 on the pre-quantization ONNX, Quant Top-1 on the
deployment model; latency single-frame single-thread single-core, FPS
4-thread concurrent; CPU 8xA55@1.8GHz performance mode, BPU
1xBayes-e@1GHz):

| Model | Size | Classes | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ConvNeXt_nano | 224x224 | 1000 | 15.59 | 77.37% | 71.75% | 5.71 | 200+ |
| ConvNeXt_pico | 224x224 | 1000 | 9.04 | 77.25% | 71.03% | 3.37 | 364+ |
| ConvNeXt_femto | 224x224 | 1000 | 5.22 | 73.75% | 72.25% | 2.46 | 556+ |
| ConvNeXt_atto | 224x224 | 1000 | 3.69 | 73.25% | 69.75% | 1.96 | 732+ |

All rows are quoted from the published X5 release table. Prepare the atto artifact using [model/README.md](model/README.md); [conversion/](conversion/README.md) also documents the nano and femto PTQ configurations.

<a id="entry-points"></a>
## Entry points

- Model preparation: [model/README.md](model/README.md)
- Python runtime: [runtime/python/README.md](runtime/python/README.md)
- Conversion: [conversion/README.md](conversion/README.md)
- Evaluation: [evaluator/README.md](evaluator/README.md)

<a id="license"></a>
## License

Sample code follows the repository top-level LICENSE (Apache-2.0). The
source models are the upstream ConvNeXt distribution
([facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt));
upstream model/weights licensing is governed by that distribution.
Published artifact use follows the applicable platform release terms.
