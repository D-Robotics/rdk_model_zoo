English | [简体中文](README_cn.md)

# 3D ResNet-18 (R3D-18) Video Action Classification

<a id="overview"></a>
## Overview

R3D-18 recognizes actions in short video clips. It extends ResNet-18 with 3D convolutions to learn spatial and temporal features together, and predicts the 400 Kinetics action classes.

References: [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248), [torchvision r3d_18](https://pytorch.org/vision/main/models/generated/torchvision.models.video.r3d_18.html).

<a id="directory"></a>
## Directory structure

```text
3dresnet/
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

There are no conversion scripts, C++ runtime files, or video decoder in this sample.

<a id="support-matrix"></a>
## Support Matrix

| Variant | x5 | s100 | s100p | s600 |
| --- | --- | --- | --- | --- |
| R3D-18 / `r3d_18.hbm` | not-supported | supported | not-supported | not-supported |

| Language | Support |
| --- | --- |
| Python | supported (S100) |
| C++ | not-supported; no C++ implementation is provided |

Board execution requires an S100 board with the `hbm_runtime` Python package. The published latency record and its measurement conditions are listed in [evaluation](evaluator/README.md).

<a id="prerequisites"></a>
## Prerequisites

- Board: RDK S100 with an S-series system image providing the `hbm_runtime` Python package; choose the image and runtime version for your deployment.
- Conversion: OpenExplorer 3.5.0 on an x86 Linux host (see [conversion](conversion/README.md)). No complete export, calibration, or compile recipe is included.
- Storage: enough space for the downloaded HBM and the supplied 2.4 MB `video0.npy`.

<a id="quickstart"></a>
## Quick Start

The following is the complete explicit path. It requires an S100 board for the second command and does not download implicitly.

```bash
# cwd: repository root
bash samples/vision/3dresnet/model/download.sh s100
# expect: samples/vision/3dresnet/model/s100/r3d_18.hbm

# cwd: repository root, on an S100 board with hbm_runtime
python3 samples/vision/3dresnet/runtime/python/main.py \
  --target s100 \
  --asset-id s:3dresnet:s100/r3d_18.hbm
# expect: exit code 0 and JSON with asset_id, target, and five predictions;
# reference: the source sample reports video0.npy's Top-1 action as archery.
```

For a convenience invocation after preparation:

```bash
# cwd: samples/vision/3dresnet/runtime/python
bash run.sh --target s100 --asset-id s:3dresnet:s100/r3d_18.hbm
```

<a id="expected-results"></a>
## Expected Results

The default clip is `test_data/video0.npy`; the source sample reports its Top-1 class as `archery`. The CLI prints a JSON result of the following form:

```json
{
  "asset_id": "s:3dresnet:s100/r3d_18.hbm",
  "target": "s100",
  "clip": ".../test_data/video0.npy",
  "predictions": [
    {"class_id": 5, "score": 0.0, "label": "archery"}
  ]
}
```

Score values depend on the compiled artifact; the numbers above illustrate the schema. The actual list contains `--top-k` entries, and labels come from the 400-entry Kinetics mapping in `test_data` (the loader strips the quote characters embedded in the original label names).

The runtime reads the prepared RGB float32 clip `test_data/video0.npy` with shape `(1,3,16,112,112)`. Prepare video decoding, frame sampling, resizing and normalization before invoking it.

<a id="entry-points"></a>
## Entry Points

- Model preparation: [`model/README.md`](model/README.md) — one exact S100 HBM asset and explicit download commands.
- Python runtime: [`runtime/python/README.md`](runtime/python/README.md) — CLI and `R3D18Classifier` API.
- Conversion: [`conversion/README.md`](conversion/README.md) — source conversion notes, screenshots, and missing-recipe boundaries.
- Evaluation: [`evaluator/README.md`](evaluator/README.md) — functional reference and the published performance record.
- Runtime language: Python.

<a id="license"></a>
## License

The sample code is Apache-2.0 under the repository license, matching the Apache header of the source sample code. No publisher SHA-256 or separate model-weight license is recorded for `r3d_18.hbm`; confirm the weight license and redistribution terms with the publisher before redistribution.
