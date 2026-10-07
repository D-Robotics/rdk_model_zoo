# YOLOWorld X5 open-vocabulary detection

<a id="overview"></a>

## Overview

This sample detects user-selected words from an **offline vocabulary embedding**
JSON on the X5 BPU. The vocabulary is a required text input asset and is not a
normal label list. YOLOWorld is an open-vocabulary region detector; the upstream
algorithm reference is [YOLO-World](https://github.com/AILab-CVC/YOLO-World).
Source: X5 platform sample delivery at `ac115717197920355fc390bb04299b20e6436864`.

<a id="support-matrix"></a>
## Support-matrix

| Target | Python | C++ | Asset | Status |
| --- | --- | --- | --- | --- |
| X5 | supported | not provided | `x5:yoloworld:yolo_world.bin` | supported |
| S100/S100P/S600 | no published asset | not provided | none | unsupported |

<a id="prerequisites"></a>
## Prerequisites

Host checks use Python 3.14.7, NumPy 2.5.3, OpenCV 4.14.0 and PyYAML 6.0.3
from the repository `.venv`. Board execution needs the X5 system Python and
`hbm_runtime` matching the installed image. The model must be prepared with the
explicit download command; the required vocabulary
`test_data/offline_vocabulary_embeddings.json` ships with the sample.

<a id="quickstart"></a>
## Quickstart

Model preparation is explicit and may access the published archive:

```bash
bash samples/vision/yoloworld/model/download.sh --target x5
python3 samples/vision/yoloworld/runtime/python/main.py --target x5 --prompts dog
```

The second command requires a recognized X5. For a network-free contract check:

```bash
.venv/bin/python samples/vision/yoloworld/runtime/python/main.py --dry-run --target x5 --prompts dog
```

<a id="expected-results"></a>
## Expected-results

The model receives `float32[1,3,640,640]` RGB NCHW and
`float32[1,32,512,1]` text embeddings. The image is resized by longest side,
placed at the top-left of a zero canvas, and has no mean/std normalization.
The native raw outputs are logically `float32[1,8400,32]` class scores and
`float32[1,8400,4]` boxes; the source runtime may expose one terminal singleton,
which the runner preserves and post-process consumes without numeric conversion. Post-processing selects each row's best text slot,
uses score threshold `0.05`, class-wise NMS `0.45`, scales coordinates back, and
returns `boxes[N,4]`, `scores[N]`, and vocabulary `class_ids[N]`.

<a id="directory"></a>
## Directory

`model/` handles explicit asset preparation. `runtime/python/` owns the
three-stage task, lazy runtime and CLI. `conversion/` records the model protocol
and conversion gaps. `evaluator/` provides the parity comparison tool.
`test_data/` contains the source image and offline vocabulary. `tests/` is
host-only and uses injected runtime fixtures.

<a id="entry-points"></a>
## Entry-points

- `runtime/python/main.py` and `runtime/python/run.sh`: inference CLI.
- `model/download.py` and `model/download.sh`: explicit model preparation.
- `evaluator/compare.py`: implementation parity comparison; it never downloads.
- Python API: `YOLOWorldTask.preprocess`, `infer`, `postprocess`, `predict`
  (`pre_process`/`forward`/`post_process` are compatibility aliases).

<a id="license"></a>
## License

The sample code is Apache-2.0 under the repository license, carrying the source
headers of the X5 platform delivery (`platforms/x5/samples/vision/yoloworld`).
