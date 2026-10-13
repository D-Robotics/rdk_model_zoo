English | [简体中文](README_cn.md)

# Faster R-CNN Python runtime

<a id="overview"></a>
## Overview

Run COCO detection on images or ByteTrack person tracking on video. `main.py` constructs the detector and calls `predict`.

<a id="directory"></a>
## Directory structure

```text
main.py       # Entry point
cli.py        # Options, model selection and image/video IO
detect.py     # Model initialization and prediction stages
tracking.py   # Person tracking
run.sh        # Shell entry
```

<a id="environment"></a>
## Environment

Use the board image’s `hbm_runtime`, NumPy, OpenCV and SciPy. X5/S100/S100P use Python 3.10; S600 uses Python 3.12. For person tracking, create a virtual environment from the repository root and install the two tracker packages. Source builds also need a C++ compiler and the installed NumPy headers.
```bash
python3 -m venv --system-site-packages .venv-fasterrcnn
. .venv-fasterrcnn/bin/activate
python3 -m pip install Cython setuptools wheel PyYAML==6.0.3
python3 -m pip install --no-build-isolation --no-deps lap==0.5.12 cython-bbox==0.1.5
```

<a id="usage"></a>
## Run

Install the target's pair under `model/<target>/` by downloading it or compiling it; see the [model guide](../../model/README.md#preparation). The default command uses the local pair recorded in the target descriptor. For a pair compiled with your own recipe, pass both paths explicitly. From the repository root:
```bash
python3 samples/vision/fasterrcnn/model/download.py --target x5
cd samples/vision/fasterrcnn/runtime/python
python3 main.py
python3 main.py --target x5 \
  --parta-path ../../model/x5/frcnn_mbv3_320_parta.bin \
  --partb-path ../../model/x5/frcnn_mbv3_320_partb.bin \
  --input ../../test_data/bus.jpg --save-path result.jpg
python3 main.py --target x5 \
  --parta-path ../../model/x5/frcnn_mbv3_320_parta.bin \
  --partb-path ../../model/x5/frcnn_mbv3_320_partb.bin \
  --video ../../../bytetrack/test_data/track_test.mp4 \
  --save-path people.mp4 --records people.tsv
```
`bash run.sh` accepts the same runtime options. On S boards select `s100`, `s100p` or `s600` and use that target’s `.hbm` files.

<a id="parameters"></a>
## Options

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--target` | `str` | `auto` | `auto`, `x5`, `s100`, `s100p`, `s600` |
| `--parta-path` | `Path` | `None` | PartA file |
| `--partb-path` | `Path` | `None` | PartB file, together with PartA |
| `--input` | `Path` | `samples/vision/fasterrcnn/test_data/bus.jpg` | Image, or video with `--track` |
| `-i` | `Path` | `samples/vision/fasterrcnn/test_data/bus.jpg` | Alias for `--input` |
| `--video` | `Path` | `None` | Video input; person tracking |
| `--track` | `bool` | false | Track input video |
| `--reset-every` | `int` | `None` | Positive frame interval |
| `--records` | `Path` | `None` | Track TSV path |
| `--no-encode` | `bool` | false | Skip output video encoding |
| `--save-path` | `Path` | `samples/vision/fasterrcnn/test_data/result.jpg` | Output path, `.mp4` for video |
| `--threads` | `int` | `None` | Image conversion threads: 1, 2, 4 |
| `--label-file` | `Path` | `samples/vision/fasterrcnn/model/coco91_labels.txt` | COCO label file |
| `--list-models` | `bool` | false | List targets and local model paths |
| `--dry-run` | `bool` | false | Print selected model paths |

`None` means the option is omitted. Model paths are then selected from the target directory; OpenCV keeps its default thread count. Default file paths above are shown relative to the repository root. Use `--help` or `-h` for command help.

<a id="results"></a>
## Outputs

Image JSON includes `boxes` (N×4 original-pixel xyxy), `scores` (N) and `labels` (N original COCO IDs), plus artifact hashes and `result_path`. The label file has 91 slots including background 0 and unused COCO IDs. Video JSON reports frame count, mean prediction/tracker times and paths. TSV columns are `frame, track_id, x1, y1, x2, y2, score`; `frame` starts at 0. `--no-encode` sets `result_path` to null.

<a id="integration-example"></a>
## Use in an application

With the X5 pair installed under `model/x5/` (downloaded or self-compiled), run this from the repository root in the board Python environment:
```python
from pathlib import Path
import cv2
from samples.vision.fasterrcnn.runtime.python.cli import resolve_selection
from samples.vision.fasterrcnn.runtime.python.detect import FasterRCNNTask

root = Path.cwd()  # repository root
model_dir = root / "samples/vision/fasterrcnn/model/x5"
selection = resolve_selection(
    "x5", parta_path=model_dir / "frcnn_mbv3_320_parta.bin",
    partb_path=model_dir / "frcnn_mbv3_320_partb.bin")
model = FasterRCNNTask(selection)
image = cv2.imread(str(root / "samples/vision/fasterrcnn/test_data/bus.jpg"))
if image is None:
    raise FileNotFoundError("bus.jpg")
result = model.predict(image)
print(result.boxes, result.scores, result.labels)

prepared = model.preprocess(image)
outputs = model.infer(prepared)
result = model.postprocess(outputs, prepared.context)
```
For ordered video frames, wrap the detector with `PersonTrackerTask(model)` from `tracking.py` and call its `predict(image)` once per frame. Call `reset()` when starting a new video.

<a id="stage-io"></a>
## Prediction stages

| Method | Input → output |
| --- | --- |
| `preprocess` | HWC BGR uint8 image → `PreparedInput` (320×320 NV12 tensors and letterbox context) |
| `infer` | Prepared input → PartA float32 objectness (3375), deltas (3375×4), feat0 (1×256×10×10) |
| `postprocess` | PartA outputs + context → CPU proposals/RoIAlign, PartB inference, final `DetectionResult` |
| `predict` | BGR image → owned boxes/scores/labels in original coordinates |

PartB takes float32: X5 `(150,256,7,7)`, S `(1,256,1050,7)`. Reuse one model instance for sequential calls.

<a id="troubleshooting"></a>
## Model and input setup

Use `--list-models` to inspect local paths. If either file is missing, download the pair for the chosen target with the [model downloader](../../model/README.md#preparation) or compile it. A descriptor hash error requires checking the file source; pass both paths explicitly for your own compiled pair. If `auto` cannot identify the board, specify `--target`. For tracker import errors, activate the environment containing `lap` and `cython_bbox`.

