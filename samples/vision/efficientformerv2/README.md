# EfficientFormerV2 image classification

EfficientFormerV2 ImageNet-1k classification on RDK X5: one BGR image in,
a stable Top-K of `(class id, score, label)` out. The X5 release ships the
S0, S1, and S2 variants (paper [EfficientFormerV2: Rethinking Vision
Transformers for MobileNet Size and
Speed](https://arxiv.org/abs/2212.08059)).
[中文说明](README_cn.md)

<a id="overview"></a>

## Overview

The sample provides a Python runtime for X5. The
`EfficientFormerV2Classifier` class runs a `preprocess → infer →
postprocess` flow chained by `predict`: it resolves one exact artifact
reference from the platform release manifest, verifies the board identity,
loads `hbm_runtime` lazily, and returns a typed Top-K result
([runtime/python/README.md](runtime/python/README.md)).

### Algorithm background

EfficientFormerV2 revisits vision transformers at MobileNet size and
speed: a fine-grained joint search optimizes latency, parameters, and
accuracy together; unified feed-forward networks, improved MHSA
(talking-head attention with locality), and attention at higher
resolutions with cheaper downsampling reduce the attention and
downsampling overhead relative to the EfficientFormer baseline while
keeping MobileNet-level size and speed
([paper](https://arxiv.org/abs/2212.08059),
[snap-research/EfficientFormer](https://github.com/snap-research/EfficientFormer)).

Feature summary:

- **Mobile-oriented backbone**: hybrid backbone for efficient image classification on edge devices.
- **Joint search strategy**: latency and parameter count optimized together when selecting architectures.
- **Hierarchical design**: four stages with feature sizes of `1/4`, `1/8`, `1/16`, and `1/32` of the input resolution.
- **Edge deployment**: S0, S1, and S2 RDK X5 deployment models with packed NV12 input.

![EfficientFormerV2 architecture](./test_data/EfficientFormerV2_architecture.png)

*Network architectures (Figure 2 of the paper): (a) the EfficientFormer
baseline network, (b) unified FFN, (c) improved MHSA, (d)(e) attention on
higher resolution, and (f) attention downsampling.*

<a id="directory"></a>
## Directory structure

```text
efficientformerv2/
├── conversion/  # Export and quantization configuration
├── evaluator/  # Evaluation commands and metrics
├── model/  # Model files and download scripts
├── runtime/  # Python and native inference implementations
├── test_data/  # Example inputs
├── tests/  # Automated tests
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
└── requirements-host.txt  # Source or data file
```

<a id="support-matrix"></a>
## Support matrix

| Target | Variant | Language | Status |
| --- | --- | --- | --- |
| x5 | s0, s1, s2 | python | supported |
| s100 / s100p / s600 | any | python | not-supported (the S manifest publishes no EfficientFormerV2 asset; selection is an explicit error, no cross-platform fallback) |

<a id="prerequisites"></a>
## Prerequisites

Board Python execution needs the board image's matching `hbm_runtime`,
NumPy, and OpenCV-Python; the SDK is imported only when a model executes
(`--help`, `--list-models`, `--dry-run`, and host tests need no SDK).
Manifest reading also needs PyYAML. On a development host, install the
user-space dependencies from `requirements-host.txt`:

```bash
# cwd: repository root — success: "host dependencies: ok"
python3 -m venv .venv-efficientformerv2
source .venv-efficientformerv2/bin/activate
python3 -m pip install -r samples/vision/efficientformerv2/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## Quick start

One complete path on an X5 board, commands run from the repository root.
Prerequisite: the board image with `hbm_runtime` and network access to the
manifest's model server.

```bash
# 1. Prepare the artifact (input: manifest row x5:efficientformerv2:EfficientFormerv2_s0_224x224_nv12.bin)
#    output: samples/vision/efficientformerv2/model/EfficientFormerv2_s0_224x224_nv12.bin
#    success: downloader exits 0 and prints the observed digest
bash samples/vision/efficientformerv2/model/download.sh x5 s0

# 2. Run classification (input: the artifact above plus the bundled test image)
#    output: Top-5 class ids, scores, labels on stdout
#    success: exit code 0 and a printed Top-5 list
python3 samples/vision/efficientformerv2/runtime/python/main.py \
  --target x5 \
  --asset-id x5:efficientformerv2:EfficientFormerv2_s0_224x224_nv12.bin \
  --model-path samples/vision/efficientformerv2/model/EfficientFormerv2_s0_224x224_nv12.bin \
  --test-img samples/vision/efficientformerv2/test_data/goldfish.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

`s1`/`s2` substitute their own references and paths (see `--list-models`);
the default variant (when none is given) is `s0`. Full commands:
[runtime/python/README.md](runtime/python/README.md).

<a id="expected-results"></a>
## Expected results

The Python run prints a stable Top-K (default 5) of class IDs, scores, and
labels and exits 0; Pass `--img-save-path` to save a visualization; otherwise results are printed to stdout. With the bundled `goldfish.JPEG` the Top-5 contains a
goldfish-related ImageNet class. Select a target and variant listed in the [Support matrix](#support-matrix), prepare that exact manifest artifact with the model downloader, and run the sample on the matching board.

<a id="performance"></a>
## Performance data

Published performance on RDK X5 (X5 release x5-v1.1.3; Float Top-1
on the pre-quantization ONNX, Quant Top-1 on the deployment model, latency
single-frame single-thread single-core, FPS multi-threaded):

| Model | Size | Params (M) | Float Top-1 | Quant Top-1 | Single-thread Latency (ms) | Multi-thread Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientFormerV2-S2 | 224x224 | 12.6 | 77.50% | 70.75% | 6.99 | 26.01 | 152.40 |
| EfficientFormerV2-S1 | 224x224 | 6.1 | 77.25% | 68.75% | 4.24 | 14.35 | 275.95 |
| EfficientFormerV2-S0 | 224x224 | 3.5 | 74.25% | 68.50% | 5.79 | 19.96 | 198.45 |

![Inference result](./test_data/inference.png)

*Reference inference result from the X5 release: the bundled
[goldfish.JPEG](test_data/goldfish.JPEG) ranks `goldfish` first, followed
by tench, axolotl, rock beauty, and coral reef.*

<a id="entry-points"></a>
## Entry points

- Model preparation: [model/README.md](model/README.md)
- Python runtime: [runtime/python/README.md](runtime/python/README.md)
- Conversion: [conversion/README.md](conversion/README.md)
- Evaluation: [evaluator/README.md](evaluator/README.md)

<a id="license"></a>
## License

Sample code follows the repository top-level LICENSE (Apache-2.0). The
source models are the upstream EfficientFormerV2 distribution
([snap-research/EfficientFormer](https://github.com/snap-research/EfficientFormer));
upstream model/weights licensing is governed by that distribution.
Published artifact use follows the applicable platform release terms.
