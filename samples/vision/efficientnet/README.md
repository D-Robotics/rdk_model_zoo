# EfficientNet image classification

EfficientNet ImageNet-1k classification on RDK boards: one BGR image in, a
stable Top-K of `(class id, score, label)` out. The X5 side ships the
EfficientNet B2/B3/B4 variants (paper [EfficientNet: Rethinking Model Scaling
for Convolutional Neural Networks](https://arxiv.org/abs/1905.11942)); the
S100/S600 side ships the EfficientNet-Lite lite0..lite4 family (the
[TensorFlow TPU EfficientNet-Lite](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
implementation, as cited by the source delivery). [中文说明](README_cn.md)

<a id="overview"></a>
## Overview

The maintained implementation is one Python flow (all targets, no C++
runtime exists for this sample on either source branch). Python resolves one
exact artifact reference from the platform release manifests, verifies the
board identity, loads `hbm_runtime` lazily, and runs a
`pre_process → forward → post_process` task
([runtime/python/README.md](runtime/python/README.md)).
The former platform branch entries remain compatibility shims under
`platforms/{x5,s}/` until the migration closeout; their audit record
lives in the migration documents, not here.

<a id="support-matrix"></a>
## Support matrix

| Target | Variant | Language | Status |
| --- | --- | --- | --- |
| x5 | b2, b3, b4 | python | supported (source-verified contract; board smoke pending, see below) |
| s100 | lite0..lite4 | python | supported (source-verified contract; board smoke pending, see below) |
| s600 | lite0..lite4 | python | supported (source-verified contract; board smoke pending, see below) |
| s100p | any | python | not-supported (no s100p asset row in the release manifest; selection is an explicit error, no fallback) |

Source baselines: X5 rdk_x5 @ac11571 (x5-v1.1.3); S rdk_s @380e1a2
(s-v1.1.2). The unified sample's host tests (25) all pass. Board smoke for
this batch is executed after the host side of all four B2 samples lands;
this matrix is updated with the observed results then — until that entry
exists, board status for this sample is **not-run**, and the legacy sources
remain the verified delivery.

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
labels and exits 0; no output files are written unless `--img-save-path` is
given. With the bundled `Scottish_deerhound.JPEG` the Top-5 contains a
deerhound-related ImageNet class; with `redshank.JPEG` a redshank-related
class. A board that cannot be identified, or a target without a matching
artifact, exits with an error instead of guessing — in particular the legacy
S behavior of silently running the lite0 S100 model on every non-S600 SoC
(including S100P) is gone: S100P is an explicit no-published-asset error.

<a id="performance"></a>
## Performance data

Published records, not re-measured in this repository.

X5 (rdk_x5 @ac11571, x5-v1.1.3):

| Model | Size | Params (M) | Float Top-1 | Quant Top-1 | Single-thread Latency (ms) | Multi-thread Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientNet-B4 | 224x224 | 19.27 | 74.25% | 71.75% | 5.44 | 18.63 | 212.75 |
| EfficientNet-B3 | 224x224 | 12.19 | 76.22% | 74.05% | 3.96 | 12.76 | 310.30 |
| EfficientNet-B2 | 224x224 | 9.07 | 76.50% | 73.25% | 3.31 | 10.51 | 376.77 |

S-series (rdk_s @380e1a2, s-v1.1.2):

| Variant | Single-thread latency | Single-thread FPS | Multi-thread latency | Multi-thread FPS |
| --- | --- | --- | --- | --- |
| Lite0 | 0.448 ms | 2107.815 | 0.591 ms | 4827.886 |
| Lite1 | 0.489 ms | 1948.957 | 0.708 ms | 4086.470 |
| Lite2 | 0.565 ms | 1702.519 | 0.935 ms | 3123.682 |
| Lite3 | 0.668 ms | 1451.031 | 1.249 ms | 2345.518 |
| Lite4 | 0.915 ms | 1064.339 | 1.979 ms | 1487.055 |

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
