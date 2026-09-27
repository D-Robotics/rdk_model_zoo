English | [简体中文](./README_cn.md)

# Python Inference

## Environment and files

Run inference on an S100 or S100P board with the board-provided `hbm_runtime`,
NumPy, OpenCV and SciPy. Torch and ONNX Runtime are not required, and the runner
does not install packages automatically.

The relevant sample layout is:

```text
yoloe26_seg/
├── model/
│   ├── nash-e/              # S100 HBM, JSON metadata and names
│   └── nash-m/              # S100P HBM, JSON metadata and names
├── runtime/python/
│   ├── main.py              # Image command-line entry point
│   ├── run.sh               # Download-and-run wrapper
│   └── yoloe26seg.py        # Staged Python API
└── test_data/
    └── office_desk.jpg      # Default input image
```

Bundled input and default model paths are resolved from the sample directory,
independent of the current working directory. Relative output paths are resolved
from the directory in which the command is invoked.

## Run

From the `samples/vision/yoloe26_seg/` sample root:

```bash
bash runtime/python/run.sh --size n
bash runtime/python/run.sh --size x --test-img /path/image.jpg --output result.jpg
python3 runtime/python/main.py --size s --march nash-m --output result.jpg
```

`run.sh` detects the board, downloads the selected model when `--model-path` is
not supplied, and verifies the published hashes. Direct `main.py` execution
expects the artifacts to already exist under `model/<march>/`, unless explicit
paths are supplied. A requested march and the model metadata must match the
detected board; S600 and other boards are rejected.

## Command-line options

| Option | Default | Description |
|---|---|---|
| `--size {n,s,m,l,x}` | `n` | Released model size. It must match the metadata. |
| `--march {nash-e,nash-m}` | Detected board | Target architecture: `nash-e` for S100 or `nash-m` for S100P. |
| `--model-path PATH` | `model/<march>/<size-specific HBM>` | HBM file. Passing it also prevents `run.sh` from downloading a model. |
| `--metadata PATH` | `model/<march>/yoloe_26<SIZE>_seg_pf.json` | JSON metadata matching the HBM. |
| `--test-img PATH` | `test_data/office_desk.jpg` | Input image decoded by OpenCV as BGR. |
| `--output PATH` | `result.jpg` | Visualization with masks, boxes, class names and scores. |
| `--json-output PATH` | None | Optional JSON array containing each detection's `box`, `score`, `class_id` and `name`; masks are not written. |
| `--score-thres FLOAT` | `0.25` | Minimum sigmoid confidence; must be strictly between 0 and 1. |
| `--max-det INTEGER` | `300` | Maximum retained detections; must be from 1 through 8400. |
| `--multi-label` | Disabled | Retain multiple classes at one candidate anchor. |

Single-label top-k selection is the default. Both selection modes omit NMS.
Preprocessing uses a centered 640x640 letterbox with padding value 114.

## Python API

`YoloE26Seg` supports both a staged interface and a one-call interface. Run an
importing script from `runtime/python/`, or add that directory to `PYTHONPATH`:

```python
import cv2

from yoloe26seg import (
    SAMPLE,
    YoloE26Seg,
    YoloE26SegConfig,
    detect_march,
    hbm_name,
    model_stem,
)

size = "n"
march = detect_march()
model_dir = SAMPLE / "model" / march
model = YoloE26Seg(YoloE26SegConfig(
    model_path=str(model_dir / hbm_name(size, march)),
    metadata_path=str(model_dir / f"{model_stem(size)}.json"),
))
image = cv2.imread(str(SAMPLE / "test_data" / "office_desk.jpg"))

input_tensor = model.pre_process(image, image_format="BGR")
raw_outputs = model.forward(input_tensor)
boxes, scores, labels, masks = model.post_process(raw_outputs, image.shape)

# These run the same three stages internally.
boxes, scores, labels, masks = model.predict(image)
boxes, scores, labels, masks = model(image)
```

The staged values have these structures:

```text
pre_process(...) -> {model_name: {y_input_name: Y_plane, uv_input_name: UV_plane}}
                     Y_plane:  (1, 640, 640, 1)
                     UV_plane: (1, 320, 320, 2)

forward(...)     -> {model_name: {output_name: raw_runtime_tensor, ...}}
```

`forward()` returns the nested dictionary from `HB_HBMRuntime.run()` unchanged.
`post_process()` performs output dequantization and returns aligned original-image
xyxy boxes, float scores, integer class IDs and a list of box-local masks. Each
mask is a `uint8` 0/1 array for `image[y1:y2, x1:x2]`; the bounds are the clipped
box coordinates converted with integer truncation, matching the shared
`draw_masks` helper. A degenerate clipped box retains an empty mask so result
indices remain aligned.

`set_scheduling_params(priority=..., bpu_cores=...)` forwards supplied scheduling
values to the runtime. Omitted values are left unchanged.
