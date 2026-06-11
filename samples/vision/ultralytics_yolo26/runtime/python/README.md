English | [简体中文](./README_cn.md)

# YOLO26 Python Inference Sample

This sample demonstrates how to run YOLO26 task models on RDK S100/S100P with `hbm_runtime`.

## Environment Dependencies

This sample has no special extra dependencies. Make sure the RDK S100/S100P Python environment is ready.

```bash
pip install numpy opencv-python hbm-runtime scipy
```

## Directory Structure

```text
.
├── main.py          # Unified inference entry script
├── yolo26_det.py    # Detection wrapper
├── yolo26_seg.py    # Instance segmentation wrapper
├── yolo26_pose.py   # Pose estimation wrapper
├── yolo26_cls.py    # Image classification wrapper
├── yolo26_obb.py    # Oriented bounding box wrapper
├── run.sh           # One-click execution script
└── README.md        # Usage instructions
```

## Parameter Description

| Parameter | Description | Default Value |
|---|---|---|
| `--task` | Task type: `detect`, `seg`, `pose`, `cls`, `obb` | (required) |
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
| `--angle-sign` | Angle decoding sign multiplier (OBB) | `1.0` |
| `--angle-offset`| Angle decoding offset (OBB) | `0.0` |

> **Note**: The default `--model-path` is determined automatically based on `--task` and the detected SoC. S100 uses the `nashe` suffix; S100P uses the `nashm` suffix.

## Quick Run

- **One-click Execution Script**
  ```bash
  bash run.sh detect
  ```

- **Manual Execution**
  - Run detection
    ```bash
    python3 main.py --task detect
    ```
  - Run with explicit parameters
    ```bash
    python3 main.py \
        --task detect \
        --model-path ../../model/nash-e/yolo26n_detect_nashe_640x640_nv12.hbm \
        --test-img ../../test_data/bus.jpg
    ```

## Task Examples

### Oriented Bounding Box (OBB)

```bash
python3 main.py \
    --task obb \
    --model-path ../../model/nash-e/yolo26n_obb_nashe_640x640_nv12.hbm \
    --test-img ../../test_data/ships-detection-using-obb.jpg \
    --label-file ../../../../../datasets/dotav1/dota_classes.names
```

### Segmentation

```bash
python3 main.py \
    --task seg \
    --model-path ../../model/nash-e/yolo26n_seg_nashe_640x640_nv12.hbm \
    --test-img ../../test_data/bus.jpg
```

## Interface Description

- **`YOLO26Config`**: Shared configuration class for all tasks.
- **`YOLO26*`**: Task-specific wrappers (Detect, Seg, Pose, Cls, OBB).

Each wrapper provides:
- `pre_process(img)`
- `forward(input_tensors)`
- `post_process(outputs)`
- `predict(img)` (high-level entry)

### Return Formats

| Task   | Return Type                                                          |
|--------|----------------------------------------------------------------------|
| detect | `(boxes, scores, cls_ids)` — numpy arrays `(N,4)`, `(N,)`, `(N,)`   |
| seg    | `(boxes, scores, cls_ids, masks)` — arrays + list of cropped binary masks |
| pose   | `(boxes, scores, cls_ids, keypoints)` — arrays + keypoints `(N,17,3)` |
| cls    | `[(class_id, probability), ...]` — list of tuples                    |
| obb    | `[{'rrect': (cx,cy,w,h,angle), 'score': float, 'id': int}, ...]`    |

Refer to the shared utilities under `utils/py_utils/` for common pre-processing, post-processing, and visualization helpers.

## Code Documentation

For detailed API references and code-level documentation, refer to the [source reference guide](../../../../docs/source_reference/README.md).
