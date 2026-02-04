# YOLO26 Python Runtime

This directory contains the Python implementation for YOLO26 inference on RDK platforms.

## Directory Structure
```
.
├── main.py          # Main entry point for all tasks
├── run.sh           # One-click execution script
├── yolo26_det.py    # Detection model implementation
├── yolo26_seg.py    # Segmentation model implementation
├── yolo26_pose.py   # Pose estimation model implementation
├── yolo26_obb.py    # OBB model implementation
├── yolo26_cls.py    # Classification model implementation
└── README.md        # This file
```

## Quick Run
The easiest way to run the sample is using the `run.sh` script, which handles model downloading and parameter configuration automatically.

```bash
cd samples/vision/yolo26/runtime/python
sh run.sh
```

## Manual Usage

### 1. Install Dependencies
Ensure you have the required libraries installed (e.g., `hobot_dnn`, `opencv-python`, `numpy`).

### 2. Download Models
Navigate to the model directory and run the download script:
```bash
cd ../../model
# Download only nano models (Fast)
sh download_model.sh
# OR Download all models
# sh fulldownload.sh
```

### 3. Run Inference
Use `main.py` to run inference. You can specify the task, model path, and input image.

**Common Arguments:**
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--task` | str | `detect` | Task type: `detect`, `seg`, `pose`, `cls`, `obb` |
| `--model-path` | str | Required | Path to the quantized `.bin` model file |
| `--test-img` | str | `bus.jpg` | Path to the input image |
| `--img-save-path` | str | `result.jpg` | Path to save the output visualization |
| `--classes-num` | int | 80 | Number of classes |
| `--score-thres` | float | 0.25 | Confidence threshold |
| `--nms-thres` | float | 0.7 | NMS IoU threshold |

**Example: Object Detection**
```bash
python3 main.py \
    --task detect \
    --model-path ../../model/yolo26n_detect_bayese_640x640_nv12.bin \
    --test-img ../../../../../datasets/coco/assets/bus.jpg \
    --img-save-path ../../test_data/result_detect.jpg
```

**Example: Instance Segmentation**
```bash
python3 main.py \
    --task seg \
    --model-path ../../model/yolo26n_seg_bayese_640x640_nv12.bin \
    --test-img ../../../../../datasets/coco/assets/bus.jpg \
    --img-save-path ../../test_data/result_seg.jpg
```

**Example: Pose Estimation**
```bash
python3 main.py \
    --task pose \
    --model-path ../../model/yolo26n_pose_bayese_640x640_nv12.bin \
    --test-img ../../../../../datasets/coco/assets/bus.jpg \
    --img-save-path ../../test_data/result_pose.jpg
```
