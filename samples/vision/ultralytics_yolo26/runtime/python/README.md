# YOLO26 Python Sample

This sample demonstrates how to run YOLO26 task models on RDK X5 with `hbm_runtime`.

## Environment Dependencies

This sample has no special extra dependencies. Make sure the RDK X5 Python environment is ready.

```bash
pip install numpy opencv-python hbm-runtime scipy
```

## Directory Structure

```text
.
├── main.py                # Inference entry script
├── yolo26_det.py          # Detection wrapper
├── yolo26_seg.py          # Segmentation wrapper
├── yolo26_pose.py         # Pose wrapper
├── yolo26_obb.py          # OBB wrapper
├── yolo26_cls.py          # Classification wrapper
├── run.sh                 # One-click execution script
└── README.md              # Usage instructions
```

## Parameter Description

| Parameter | Description | Default Value |
|---|---|---|
| `--task` | Task type: `detect`, `seg`, `pose`, `cls`, `obb` | `detect` |
| `--model-path` | Path to the RDK X5 `.bin` model file | `../../model/yolo26n_detect_bayese_640x640_nv12.bin` |
| `--test-img` | Path to the test input image | `../../../../../datasets/coco/assets/bus.jpg` |
| `--label-file` | Path to the label file, empty means using task default | `""` |
| `--img-save-path` | Path to save the result image | `../../test_data/result_detect.jpg` |
| `--priority` | Model priority | `0` |
| `--bpu-cores` | BPU core indexes used for inference | `0` |
| `--classes-num` | Number of classes used by detection-style tasks | `80` |
| `--score-thres` | Score threshold | `0.25` |
| `--nms-thres` | NMS threshold | `0.70` |
| `--strides` | Decoder strides | `8,16,32` |
| `--topk` | Top-K results for classification | `5` |
| `--kpt-conf-thres` | Keypoint confidence threshold for pose | `0.50` |
| `--angle-sign` | Angle sign for OBB decoding | `1.0` |
| `--angle-offset` | Angle offset for OBB decoding | `0.0` |
| `--regularize` | Whether to regularize OBB angle | `1` |

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
        --model-path ../../model/yolo26n_detect_bayese_640x640_nv12.bin \
        --test-img ../../../../../datasets/coco/assets/bus.jpg \
        --img-save-path ../../test_data/result_detect.jpg
    ```

## Task Examples

### Segmentation

```bash
python3 main.py \
    --task seg \
    --model-path ../../model/yolo26n_seg_bayese_640x640_nv12.bin \
    --test-img ../../../../../datasets/coco/assets/bus.jpg \
    --img-save-path ../../test_data/result_seg.jpg
```

### Pose

```bash
python3 main.py \
    --task pose \
    --model-path ../../model/yolo26n_pose_bayese_640x640_nv12.bin \
    --test-img ../../../../../datasets/coco/assets/bus.jpg \
    --img-save-path ../../test_data/result_pose.jpg
```

### OBB

```bash
python3 main.py \
    --task obb \
    --model-path ../../model/yolo26n_obb_bayese_640x640_nv12.bin \
    --test-img ../../../../../datasets/dotav1/asset/P0009.png \
    --label-file ../../../../../datasets/dotav1/dota_classes.names \
    --img-save-path ../../test_data/result_obb.jpg
```

### Classification

```bash
python3 main.py \
    --task cls \
    --model-path ../../model/yolo26n_cls_bayese_224x224_nv12.bin \
    --test-img ../../../../../datasets/imagenet/asset/zebra_cls.jpg \
    --label-file ../../../../../datasets/imagenet/imagenet_classes.names
```

## Interface Description

- **`YOLO26*Config`**: Encapsulates model path and runtime parameters for each task.
- **`YOLO26*`**: Provides the complete inference pipeline, including `pre_process`, `forward`, `post_process`, and `predict`.

Refer to the shared utilities under `utils/py_utils/` for common pre-processing, post-processing, and visualization helpers.
