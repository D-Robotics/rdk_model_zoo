# EdgeNeXt image classification

EdgeNeXt ImageNet-1k classification on RDK X5: one BGR image in, a
stable Top-K of `(class id, score, label)` out. The X5 release ships the
base, small, x-small, and xx-small variants (paper [EdgeNeXt: Efficiently
Amalgamated CNN-Transformer Architecture for Mobile Vision
Applications](https://arxiv.org/abs/2206.10589), reference
[mmaaz60/EdgeNeXt](https://github.com/mmaaz60/EdgeNeXt)).
[中文说明](README_cn.md)

<a id="overview"></a>

## Overview

EdgeNeXt is an efficient hybrid CNN-Transformer architecture for mobile
vision: a four-stage pyramid combines convolutional encoders with SDTA
(Split Depth-wise Transpose Attention) encoders to balance classification
accuracy, model size, and inference speed. It targets ImageNet-1k
1000-class classification. Four feature highlights:

- **Hybrid CNN-Transformer design** — combines the inference efficiency
  of convolutions with transformer-style global feature modeling.
- **Four-stage pyramid** — a deployment-friendly hierarchical feature
  extraction structure.
- **SDTA encoder** — encodes multi-scale features through channel
  splitting (split 3×3 branches) and transpose attention.
- **Efficient deployment** — base, small, x-small, and xx-small RDK X5
  deployment models with packed NV12 input.

![EdgeNeXt architecture: four-stage pyramid with NxN convolution encoder
and SDTA encoder details](./test_data/EdgeNeXt_architecture.png)

*Figure: the EdgeNeXt architecture — the four-stage pyramid (top) with
the NxN convolution encoder (bottom left) and the SDTA encoder with its
split 3×3 branches and transpose attention (bottom right). It depicts the
upstream training architecture, while the deployed artifacts are the
INT8-quantized base/small/x_small/xx_small variants at 224×224 NV12 (see
[Support matrix](#support-matrix)).*

The sample provides a Python runtime for X5. The `EdgeNeXtClassifier` class runs a `preprocess → infer → postprocess` flow chained by `predict`: it resolves one exact artifact reference from the platform release manifest, verifies the board identity, loads `hbm_runtime` lazily, and returns a typed Top-K result
([runtime/python/README.md](runtime/python/README.md)).

<a id="directory"></a>
## Directory structure

```text
edgenext/
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
| x5 | base, small, x_small, xx_small | python | supported |
| s100 / s100p / s600 | any | python | not-supported (the S manifest publishes no EdgeNeXt asset; selection is an explicit error, no cross-platform fallback) |

<a id="prerequisites"></a>
## Prerequisites

Board Python execution needs the board image's matching `hbm_runtime`,
NumPy, and OpenCV-Python; the SDK is imported only when a model executes
(`--help`, `--list-models`, `--dry-run`, and host tests need no SDK).
Manifest reading also needs PyYAML. On a development host, install the
user-space dependencies from `requirements-host.txt`:

```bash
# cwd: repository root — success: "host dependencies: ok"
python3 -m venv .venv-edgenext
source .venv-edgenext/bin/activate
python3 -m pip install -r samples/vision/edgenext/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## Quick start

One complete path on an X5 board, commands run from the repository root.
Prerequisite: the board image with `hbm_runtime` and network access to the
manifest's model server.

```bash
# 1. Prepare the artifact (input: manifest row x5:edgenext:EdgeNeXt_base_224x224_nv12.bin)
#    output: samples/vision/edgenext/model/EdgeNeXt_base_224x224_nv12.bin
#    success: downloader exits 0 and prints the observed digest
bash samples/vision/edgenext/model/download.sh x5 base

# 2. Run classification (input: the artifact above plus the bundled test image)
#    output: Top-5 class ids, scores, labels on stdout
#    success: exit code 0 and a printed Top-5 list
python3 samples/vision/edgenext/runtime/python/main.py \
  --target x5 \
  --asset-id x5:edgenext:EdgeNeXt_base_224x224_nv12.bin \
  --model-path samples/vision/edgenext/model/EdgeNeXt_base_224x224_nv12.bin \
  --test-img samples/vision/edgenext/test_data/Zebra.jpg \
  --label-file datasets/imagenet/imagenet_classes.names
```

`small`/`x_small`/`xx_small` substitute their own reference and path
(see `--list-models`); the default variant (when none is given) is
`base`. Full commands:
[runtime/python/README.md](runtime/python/README.md).

<a id="expected-results"></a>
## Expected results

The Python run prints a stable Top-K (default 5) of class IDs, scores, and
labels and exits 0; Pass `--img-save-path` to save a visualization; otherwise results are printed to stdout. With the bundled `Zebra.jpg` the Top-5 contains a
zebra-related ImageNet class. Select a target and variant listed in the [Support matrix](#support-matrix), prepare that exact manifest artifact with the model downloader, and run the sample on the matching board.

The screenshot below shows a reference run from the X5 release: the demo
overlay draws the top-5 ranks onto the bundled `Zebra.jpg`, with rank 1
being class 340 (zebra).

![Reference inference result on X5: zebra test image with the top-5
overlay, rank 1 class 340 (zebra)](./test_data/inference.png)

<a id="performance"></a>
## Performance data

Published performance on RDK X5 (X5 release x5-v1.1.3; Float Top-1
on the pre-quantization ONNX, Quant Top-1 on the deployment model, latency
single-frame single-thread single-core, FPS multi-threaded):

| Model | Size | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- |
| EdgeNeXt-base | 224x224 | 18.51 | 78.21% | 74.52% | 8.80 | 113.35 |
| EdgeNeXt-small | 224x224 | 5.59 | 76.50% | 71.75% | 4.41 | 226.15 |
| EdgeNeXt-x-small | 224x224 | 2.34 | 71.75% | 66.25% | 2.88 | 345.73 |
| EdgeNeXt-xx-small | 224x224 | 1.33 | 69.50% | 64.25% | 2.47 | 403.49 |

<a id="entry-points"></a>
## Entry points

- Model preparation: [model/README.md](model/README.md)
- Python runtime: [runtime/python/README.md](runtime/python/README.md)
- Conversion: [conversion/README.md](conversion/README.md)
- Evaluation: [evaluator/README.md](evaluator/README.md)

<a id="license"></a>
## License

Sample code follows the repository top-level LICENSE (Apache-2.0). The
source models are the upstream EdgeNeXt distribution; upstream
model/weights licensing is governed by that distribution (see the paper
link above). Published artifacts follow the platform release manifests;
Python code is licensed under Apache-2.0. Review applicable upstream model and weight license terms for those components.
