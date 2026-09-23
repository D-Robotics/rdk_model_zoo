English | [简体中文](./README_cn.md)

# 3D ResNet-18 (R3D-18) Video Action Classification

<a id="overview"></a>
## Overview

R3D-18 classifies a preprocessed 16-frame video clip into one of the 400 Kinetics action classes. The network extends ResNet-18 with 3D convolutions so that spatial and temporal features are modeled together; the unified runtime consumes an already normalized RGB float32 NumPy clip and applies source-compatible softmax Top-K decoding.

- Paper: [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248)
- Reference implementation: [torchvision r3d_18](https://pytorch.org/vision/main/models/generated/torchvision.models.video.r3d_18.html)
- Repository location: `samples/vision/3dresnet`
- Source baseline: `platforms/s/samples/vision/3dresnet`, source inventory at `380e1a2bf42041af54be6f34935e50197cfadff9`

The input is not a video file. `test_data/video0.npy` is the prepared `(1, 3, 16, 112, 112)` clip. Frame decoding, sampling, resizing, and normalization are outside this sample.

<a id="support-matrix"></a>
## Support Matrix

| Variant | x5 | s100 | s100p | s600 |
| --- | --- | --- | --- | --- |
| R3D-18 / `r3d_18.hbm` | not-supported | supported-not-run | not-supported | not-supported |

| Language | State |
| --- | --- |
| Python | supported-not-run on S100; host fixture tests pass |
| C++ | not-supported; no C++ implementation is provided |

The S100 Python board path has not been run in this migration. Host tests do not verify `hbm_runtime`, HBM execution, latency, or board output. See [evaluation](evaluator/README.md) for the source performance record and its status.

<a id="prerequisites"></a>
## Prerequisites

- Board: RDK S100 for the published artifact; board image and `hbm_runtime` version were not recorded in the source material, so board compatibility is supported-not-run.
- Host checks: repository `.venv` with Python 3.14.7, `numpy`, and `PyYAML`; run the commands in [Python runtime](runtime/python/README.md).
- Conversion: the source notes name OpenExplorer 3.5.0 on an x86 Linux host. No complete export, calibration, or compile recipe is included.
- Storage: enough space for the downloaded HBM and the supplied 2.4 MB `video0.npy`; no further memory requirement is recorded.

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
# the source sample describes video0.npy's Top-1 action as archery.
```

For a convenience invocation after preparation:

```bash
# cwd: samples/vision/3dresnet/runtime/python
bash run.sh --target s100 --asset-id s:3dresnet:s100/r3d_18.hbm
```

<a id="expected-results"></a>
## Expected Results

The default clip is `test_data/video0.npy`. The source sample identifies its expected functional Top-1 class as `archery`; this is a source reference, not a current board measurement. The unified CLI emits JSON rather than the legacy formatted text:

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

The score above is a schema example; no numeric board score is asserted. The actual list contains `--top-k` entries, and labels come from the 400-entry JSON mapping after removing the source file's literal double-quote characters from names.

<a id="directory"></a>
## Directory Layout

```text
3dresnet/
├── conversion/                 # conversion notes and preserved source screenshots
├── model/                      # explicit manifest-backed HBM preparation
├── runtime/python/             # binding, lazy runner, task, labels, CLI, and run.sh
├── evaluator/                  # source functional/performance record and boundaries
├── test_data/                  # prepared clip, 400 labels, and source screenshots
├── requirements-host.txt       # host test dependencies
├── README.md                   # this document
└── README_cn.md                # Chinese counterpart
```

There are no conversion scripts, C++ runtime files, or video decoder in this sample.

<a id="entry-points"></a>
## Entry Points

- Model preparation: [`model/README.md`](model/README.md) — one exact S100 HBM asset and explicit download commands.
- Python runtime: [`runtime/python/README.md`](runtime/python/README.md) — CLI and four-stage `VideoClassificationTask` API.
- Conversion: [`conversion/README.md`](conversion/README.md) — source conversion notes, screenshots, and missing-recipe boundaries.
- Evaluation: [`evaluator/README.md`](evaluator/README.md) — functional reference, source performance table, and not-run status.
- C++ runtime: not provided; therefore no C++ support is claimed.

<a id="license"></a>
## License

The migrated sample code is Apache-2.0 under the repository license. The source sample code carries the same Apache license header. The manifest does not record a publisher SHA-256 or a separate model-weight license for `r3d_18.hbm`; treat the weight license and redistribution terms as unknown and obtain them from the publisher before redistribution.
