English | [简体中文](./README_cn.md)

# Ultralytics YOLO Python Runtime

This sample demonstrates how to run Ultralytics YOLO task models on RDK S100/S100P with `hbm_runtime`.

## Environment Dependencies

This sample has no special extra dependencies. Make sure the RDK S100/S100P Python environment is ready.

```bash
pip install numpy opencv-python hbm-runtime scipy
```

## Directory Structure

```text
.
├── main.py                # Inference entry script
├── yolo_detect.py         # DFL-based detection wrapper (v5u/v8/v11/v12)
├── yolo_seg.py            # Instance segmentation wrapper
├── yolo_pose.py           # Pose estimation wrapper
├── yolo_cls.py            # Classification wrapper
├── yolo_v10detect.py      # NMS-free detection wrapper (YOLOv10)
├── run.sh                 # One-click execution script
└── README.md              # Usage instructions
```

## Parameter Description

| Parameter | Description | Default Value |
|---|---|---|
| `--task` | Task type: `detect`, `seg`, `pose`, `cls` | (required) |
| `--model-path` | Path to the `.hbm` model file | Auto-determined by task |
| `--test-img` | Path to the test input image | `../../test_data/bus.jpg` |
| `--label-file` | Path to the label file, empty means using task default | `""` |
| `--img-save-path` | Path to save the result image | `result.jpg` |
| `--priority` | Model priority | `0` |
| `--bpu-cores` | BPU core indexes used for inference | `0` |
| `--score-thres` | Score threshold | `0.25` |
| `--nms-thres` | NMS threshold | `0.45` |
| `--topk` | Top-K results for classification | `5` |
| `--kpt-conf-thres` | Keypoint confidence threshold for pose | `0.50` |

> **Note**: The default `--model-path` is determined automatically based on `--task` and the detected SoC. S100 uses the `nashe` suffix; S100P uses the `nashm` suffix. Model family and scale are selected by `--model-path`.
> Public S100P models are available for `yolov5u detect`, `yolov8 detect/seg/pose/cls`, `yolov10 detect`, `yolo11 detect/seg/pose/cls`, and `yolo12 detect`.

## Quick Run

- **One-click Execution Script**
  ```bash
  bash run.sh detect
  ```

- **Manual Execution**
  - Use default parameters
    ```bash
    python3 main.py --task detect
    ```
  - Run detection with explicit parameters
    ```bash
    python3 main.py \
        --task detect \
        --model-path ../../model/nash-e/yolo11n_detect_nashe_640x640_nv12.hbm \
        --test-img ../../test_data/bus.jpg
    ```
  - Run detection explicitly on RDK S100P
    ```bash
    python3 main.py \
        --task detect \
        --model-path ../../model/nash-m/yolo11n_detect_nashm_640x640_nv12.hbm \
        --test-img ../../test_data/bus.jpg
    ```

## Task Examples

### YOLOv8 Segmentation

```bash
python3 main.py \
    --task seg \
    --model-path ../../model/nash-e/yolov8n_seg_nashe_640x640_nv12.hbm \
    --test-img ../../test_data/bus.jpg
```

### YOLOv10 Detection (NMS-free)

```bash
python3 main.py \
    --task detect \
    --model-path ../../model/nash-e/yolov10n_detect_nashe_640x640_nv12.hbm \
    --test-img ../../test_data/bus.jpg
```

### YOLO11 Pose

```bash
python3 main.py \
    --task pose \
    --model-path ../../model/nash-e/yolo11n_pose_nashe_640x640_nv12.hbm \
    --test-img ../../test_data/bus.jpg
```

### YOLO12 Detection

```bash
python3 main.py \
    --task detect \
    --model-path ../../model/nash-m/yolo12n_detect_nashm_640x640_nv12.hbm \
    --test-img ../../test_data/bus.jpg
```

## Interface Description

- **`Yolo*Config`**: Encapsulates model path and runtime parameters for each task.
- **`Yolo*`**: Provides the complete inference pipeline, including `pre_process`, `forward`, `post_process`, and `predict`.

Each wrapper class follows the standard interface:

```python
class Wrapper:
    def __init__(self, config)
    def set_scheduling_params(self, priority=None, bpu_cores=None)
    def pre_process(self, img, image_format="BGR")
    def forward(self, input_tensor)
    def post_process(self, outputs, ...)
    def predict(self, img, ...)
    def __call__(self, img, ...)
```

### Return Formats

| Task   | Return Type                                                          |
|--------|----------------------------------------------------------------------|
| detect | `(boxes, scores, cls_ids)` — numpy arrays `(N,4)`, `(N,)`, `(N,)`   |
| seg    | `(boxes, scores, cls_ids, masks)` — arrays + list of binary masks   |
| pose   | `(boxes, scores, cls_ids, kpts_xy, kpts_score)` — numpy arrays      |
| cls    | `[(class_id, probability), ...]` — list of tuples                    |

Refer to the shared utilities under `utils/py_utils/` for common pre-processing, post-processing, and visualization helpers.

## Code Documentation

For detailed API references and code-level documentation, refer to the [source reference guide](../../../../docs/source_reference/README.md).
