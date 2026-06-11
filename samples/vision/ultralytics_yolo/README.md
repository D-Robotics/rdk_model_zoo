English | [简体中文](./README_cn.md)

# Ultralytics YOLO Model Description

This directory provides the complete usage guide for the Ultralytics YOLO sample in Model Zoo, including algorithm overview, model conversion, runtime inference, model file management, and evaluation notes.

---

## Algorithm Overview

Ultralytics YOLO is a real-time vision model family covering object detection,
instance segmentation, pose estimation, and image classification. This sample
provides RDK S100/S100P deployment examples for the following public model
families:

- Detection:
  `YOLOv5u / YOLOv8 / YOLOv10 / YOLO11 / YOLO12`
- Instance Segmentation:
  `YOLOv8 / YOLO11`
- Pose Estimation:
  `YOLOv8 / YOLO11`
- Image Classification:
  `YOLOv8 / YOLO11`

YOLOv5u detect covers the public `yolov5nu/su/mu/lu/xu` model family.

- **Official Implementation**: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

### Algorithm Capabilities

- Object detection
- Instance segmentation
- Pose estimation
- Image classification

### Algorithm Features

- **Unified model-family entry**: `main.py` and `run.sh` select the model family and task.
- **Task-specific wrappers**: detection, segmentation, pose, and classification use fixed wrappers instead of output-structure guessing.
- **NV12 input**: runtime feeds Y and UV planes as two HBM input tensors.

### Platform Notes

- Target platforms: `RDK S100` / `RDK S100P`
- Runtime backend: `hbm_runtime`
- Inference model format: `.hbm`
- Input format: `NV12` (Y + UV as two separate tensors)

---

## Directory Structure

```bash
.
├── conversion/                     # Model conversion workflow
├── evaluator/                      # Accuracy and benchmark documents
├── model/                          # Model files and download scripts
│   ├── download_model.sh           # Download models by variant and task
│   └── README.md                   # Model file description
├── runtime/                        # Runtime samples
│   └── python/                     # Python runtime sample
│       ├── main.py                 # Python entry script
│       ├── yolo_detect.py          # Detection wrapper (v5u/v8/v11/v12)
│       ├── yolo_seg.py             # Segmentation wrapper
│       ├── yolo_pose.py            # Pose wrapper
│       ├── yolo_cls.py             # Classification wrapper
│       ├── yolo_v10detect.py       # NMS-free detection wrapper (v10)
│       ├── run.sh                  # One-click execution script
│       └── README.md               # Python runtime documentation
├── test_data/                      # Test images and result images
├── README.md                       # Overview document
└── README_cn.md                    # Chinese overview document
```

---

## Quick Start

For a quick experience, run the script under `runtime/python`.

### Python

```bash
cd runtime/python
bash run.sh detect
```

The default command downloads `yolo11n_detect_nashe_640x640_nv12.hbm` if
needed and saves the result image into `test_data/`.

For detailed arguments and task examples, refer to
[runtime/python/README.md](./runtime/python/README.md).

---

## Model Conversion

This sample provides pre-converted `.hbm` models for RDK S100/S100P.

- If you only want to run inference, download models from [model/README.md](./model/README.md) and skip conversion.
- If you need to export ONNX, prepare calibration data, or compile the model,
  refer to [conversion/README.md](./conversion/README.md).

---

## Runtime

The current sample provides Python runtime implementation.

### Python Version

- Uses `hbm_runtime` as the inference backend
- Uses one unified `main.py` entry for all tasks, dispatched by `--task`; the concrete model family and scale are selected by `--model-path`
- Uses `Config + Model` wrapper style for each task

For detailed usage, refer to
[runtime/python/README.md](./runtime/python/README.md).

---

## Model Evaluation

The `evaluator/` directory contains benchmark tables, accuracy references, and
runtime validation records for the supported models.

Refer to [evaluator/README.md](./evaluator/README.md) for details.

---

## Inference Result

The Python runtime provides `run.sh` coverage for the following documented
models on `RDK S100` / `RDK S100P`:

- Detect:
  `YOLOv5u / YOLOv8 / YOLOv10 / YOLO11 / YOLO12`
- Seg:
  `YOLOv8 / YOLO11`
- Pose:
  `YOLOv8 / YOLO11`
- CLS:
  `YOLOv8 / YOLO11`

This sample covers only Ultralytics YOLO families.

With the default test images, each task should produce boxes, masks, keypoints, or classification labels that match the image content. Detailed benchmark data and result-check notes are maintained in [evaluator/README.md](./evaluator/README.md).

---

## Inference Results

With the default test images, the task produces visualization results in `test_data/`.

![Detection Result](./test_data/result_detect.jpg)

---

## License

Follows the Model Zoo top-level License.
