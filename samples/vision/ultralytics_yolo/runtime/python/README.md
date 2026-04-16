# Ultralytics YOLO Python Sample

This sample demonstrates how to run Ultralytics YOLO task models on RDK X5
with `hbm_runtime`.

## Environment Dependencies

This sample has no special extra dependencies. Make sure the RDK X5 Python
environment is ready.

```bash
pip install numpy opencv-python hbm-runtime scipy
```

## Directory Structure

```text
.
|-- main.py                 # Inference entry script
|-- ultralytics_yolo_det.py # Detection wrapper
|-- ultralytics_yolo_seg.py # Segmentation wrapper
|-- ultralytics_yolo_pose.py# Pose wrapper
|-- ultralytics_yolo_cls.py # Classification wrapper
|-- run.sh                  # One-click execution script
`-- README.md               # Usage instructions
```

## Parameter Description

| Parameter | Description | Default Value |
|---|---|---|
| `--task` | Task type: `detect`, `seg`, `pose`, `cls` | `detect` |
| `--model-path` | Path to the RDK X5 `.bin` model file | `../../model/yolo11n_detect_bayese_640x640_nv12.bin` |
| `--test-img` | Path to the test input image | `../../test_data/bus.jpg` |
| `--label-file` | Path to the label file, empty means using the task default | `""` |
| `--img-save-path` | Path to save the result image for `detect`, `seg`, and `pose` | `../../test_data/result_detect.jpg` |
| `--priority` | Model priority | `0` |
| `--bpu-cores` | BPU core indexes used for inference | `[0]` |
| `--classes-num` | Number of classes used by detection-style tasks | `80` |
| `--score-thres` | Score threshold | `0.25` |
| `--nms-thres` | NMS threshold | `0.70` |
| `--strides` | Decoder strides | `8,16,32` |
| `--reg` | DFL regression channel count | `16` |
| `--mc` | Segmentation mask coefficient count | `32` |
| `--nkpt` | Number of pose keypoints | `17` |
| `--kpt-conf-thres` | Keypoint confidence threshold for pose visualization | `0.50` |
| `--topk` | Top-K results for classification | `5` |
| `--resize-type` | Resize policy, `0` for direct resize and `1` for letterbox | `1` |

## Quick Run

- **One-click Execution Script**
  ```bash
  chmod +x run.sh
  ./run.sh
  ```

- **Manual Execution**
  - Use default parameters
    ```bash
    python3 main.py
    ```
  - Run detection with explicit parameters
    ```bash
    python3 main.py \
        --task detect \
        --model-path ../../model/yolo11n_detect_bayese_640x640_nv12.bin \
        --test-img ../../test_data/bus.jpg \
        --img-save-path ../../test_data/result_detect.jpg
    ```

## Task Examples

### Segmentation

```bash
python3 main.py \
    --task seg \
    --model-path ../../model/yolo11n_seg_bayese_640x640_nv12.bin \
    --test-img ../../test_data/bus.jpg \
    --img-save-path ../../test_data/result_seg.jpg
```

### Pose

```bash
python3 main.py \
    --task pose \
    --model-path ../../model/yolo11n_pose_bayese_640x640_nv12.bin \
    --test-img ../../test_data/bus.jpg \
    --img-save-path ../../test_data/result_pose.jpg
```

### Classification

```bash
python3 main.py \
    --task cls \
    --model-path ../../model/yolo11n_cls_detect_bayese_640x640_nv12.bin \
    --test-img ../../test_data/zebra_cls.jpg \
    --label-file ../../../../../datasets/imagenet/imagenet_classes.names
```

## Interface Description

- **`UltralyticsYOLO*Config`**: Encapsulates model path and runtime parameters for each task.
- **`UltralyticsYOLO*`**: Provides the complete inference pipeline, including `pre_process`, `forward`, `post_process`, and `predict`.

Refer to the shared utilities under `utils/py_utils/` for common
pre-processing, post-processing, and visualization helpers.
