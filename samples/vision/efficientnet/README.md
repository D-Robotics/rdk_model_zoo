# EfficientNet image classification

EfficientNet ImageNet-1k classification on RDK boards: one BGR image in, a
stable Top-K of `(class id, score, label)` out. The X5 side ships the
EfficientNet B2/B3/B4 variants (paper [EfficientNet: Rethinking Model Scaling
for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)); the
S100/S600 side ships the EfficientNet-Lite lite0..lite4 family (the
[TensorFlow TPU EfficientNet-Lite](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
implementation). [中文说明](README_cn.md)

<a id="overview"></a>

## Overview

A single Python runtime serves all supported targets.
The `EfficientNetClassifier` class runs a `preprocess → infer →
postprocess` flow chained by `predict`: it resolves one exact artifact
reference from the platform release manifest for the detected board,
verifies the board identity, loads `hbm_runtime` lazily, and returns a
typed Top-K result ([runtime/python/README.md](runtime/python/README.md)).

### Algorithm background

EfficientNet balances input resolution, depth, and width through compound
scaling: instead of tuning one dimension independently, a fixed compound
coefficient scales all three together, improving accuracy under a fixed
compute budget; neural architecture search provides the efficient
baseline network ([paper](https://arxiv.org/abs/1905.11946),
[EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)).
The X5 deployment ships B2/B3/B4; the S delivery ships the
edge-oriented EfficientNet-Lite family (lite0–lite4, TensorFlow TPU
implementation), served by the same Python flow with per-variant input
geometry (224/240/260/300/380).

Feature summary:

- **Compound scaling**: jointly scales resolution, depth, and width to balance accuracy and efficiency.
- **AutoML backbone search**: neural architecture search obtains the efficient baseline network.
- **Efficient deployment**: B2/B3/B4 RDK X5 deployment models with packed NV12 input.

![Model scaling](./test_data/efficientnet_architecture.png)

*Compound scaling (Figure 2 of the paper): baseline network (a),
conventional single-dimension scaling (b)–(d), and the compound scaling
(e) that uniformly scales width, depth, and resolution with a fixed
ratio.*

<a id="support-matrix"></a>
## Support matrix

| Target | Variant | Language | Status |
| --- | --- | --- | --- |
| x5 | b2, b3, b4 | python | supported |
| s100 | lite0..lite4 | python | supported |
| s600 | lite0..lite4 | python | supported |
| s100p | any | python | not-supported (no s100p asset row in the release manifest; selection is an explicit error, no fallback) |

With the variant omitted, the default entry resolves `b2` on x5 and
`lite0` on s100/s600; the S geometry follows the variant
(lite0..lite4 = 224/240/260/300/380).

<a id="prerequisites"></a>
## Prerequisites

Board Python execution needs the board image's matching `hbm_runtime`,
NumPy, and OpenCV-Python; the SDK is imported only when a model executes
(`--help`, `--list-models`, `--dry-run`, and host tests need no SDK).
Manifest reading also needs PyYAML. On a development host, install the
user-space dependencies from `requirements-host.txt`:

```bash
# cwd: repository root — success: "host dependencies: ok"
python3 -m venv .venv-efficientnet
source .venv-efficientnet/bin/activate
python3 -m pip install -r samples/vision/efficientnet/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## Quick start

One complete path on an X5 board, commands run from the repository root.
Prerequisite: the board image with `hbm_runtime` and network access to the
manifest's model server.

```bash
# 1. Prepare the artifact (input: manifest row x5:efficientnet:EfficientNet_B2_224x224_nv12.bin)
#    output: samples/vision/efficientnet/model/EfficientNet_B2_224x224_nv12.bin
#    success: downloader exits 0 and prints the observed digest
bash samples/vision/efficientnet/model/download.sh x5 b2

# 2. Run classification (input: the artifact above plus the bundled test image)
#    output: Top-5 class ids, scores, labels on stdout
#    success: exit code 0 and a printed Top-5 list
python3 samples/vision/efficientnet/runtime/python/main.py \
  --target x5 \
  --asset-id x5:efficientnet:EfficientNet_B2_224x224_nv12.bin \
  --model-path samples/vision/efficientnet/model/EfficientNet_B2_224x224_nv12.bin \
  --test-img samples/vision/efficientnet/test_data/Scottish_deerhound.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

For S100/S600 use the matching `s:efficientnet:s…` reference (see
`--list-models`) and the same root `datasets/imagenet/` labels; the S
geometry follows the variant (lite0..lite4 = 224/240/260/300/380). Full
commands: [runtime/python/README.md](runtime/python/README.md).

<a id="expected-results"></a>
## Expected results

The Python run prints a stable Top-K (default 5) of class IDs, scores, and
labels and exits 0; Pass `--img-save-path` to save a visualization; otherwise results are printed to stdout. With the bundled `Scottish_deerhound.JPEG` the Top-5 contains a
deerhound-related ImageNet class; with `redshank.JPEG` a redshank-related
class. Select a target and variant listed in the [Support matrix](#support-matrix), prepare the matching manifest artifact, and run the sample on that board. The runtime pairs each model path with its exact target reference.

<a id="performance"></a>
## Performance data

Published performance records.

X5 (x5-v1.1.3):

| Model | Size | Params (M) | Float Top-1 | Quant Top-1 | Single-thread Latency (ms) | Multi-thread Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientNet-B4 | 224x224 | 19.27 | 74.25% | 71.75% | 5.44 | 18.63 | 212.75 |
| EfficientNet-B3 | 224x224 | 12.19 | 76.22% | 74.05% | 3.96 | 12.76 | 310.30 |
| EfficientNet-B2 | 224x224 | 9.07 | 76.50% | 73.25% | 3.31 | 10.51 | 376.77 |

S-series (s-v1.1.2):

| Variant | Single-thread latency | Single-thread FPS | Multi-thread latency | Multi-thread FPS |
| --- | --- | --- | --- | --- |
| Lite0 | 0.448 ms | 2107.815 | 0.591 ms | 4827.886 |
| Lite1 | 0.489 ms | 1948.957 | 0.708 ms | 4086.470 |
| Lite2 | 0.565 ms | 1702.519 | 0.935 ms | 3123.682 |
| Lite3 | 0.668 ms | 1451.031 | 1.249 ms | 2345.518 |
| Lite4 | 0.915 ms | 1064.339 | 1.979 ms | 1487.055 |

![Inference result](./test_data/inference.png)

*Reference inference result from the X5 release: the bundled
[redshank.JPEG](test_data/redshank.JPEG) ranks `redshank` first, followed
by ruddy turnstone, water ouzel, oystercatcher, and dowitcher.*

<a id="directory"></a>
## Directory

- [model/](model/README.md) — manifest-driven artifact download, no checked-in binaries
- [runtime/python/](runtime/python/README.md) — canonical Python entrypoint and task modules
- [conversion/](conversion/README.md) — full S-side lite recipe + X5-side reference PTQ configs
- [evaluator/](evaluator/README.md) — published benchmarks and functional checks
- `test_data/` — bundled test images ([Scottish_deerhound.JPEG](test_data/Scottish_deerhound.JPEG), [redshank.JPEG](test_data/redshank.JPEG), [zebra_cls.jpg](test_data/zebra_cls.jpg))
- `tests/` — host unittest suite

<a id="entry-points"></a>
## Entry points

- Model preparation: [model/README.md](model/README.md)
- Python runtime: [runtime/python/README.md](runtime/python/README.md)
- Conversion: [conversion/README.md](conversion/README.md)
- Evaluation: [evaluator/README.md](evaluator/README.md)

<a id="license"></a>
## License

Sample code follows the repository top-level LICENSE (Apache-2.0). The
source models are the upstream EfficientNet / EfficientNet-Lite
distributions; upstream model/weights licensing is governed by those
distributions (see the paper links above). Published artifacts follow the
platform release manifests; the manifests carry no separate license field,
and no additional license is claimed here.
