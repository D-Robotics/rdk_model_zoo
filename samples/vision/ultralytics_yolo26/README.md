English | [简体中文](./README_cn.md)

# YOLO26 Model Description

This directory provides the complete usage guide for the YOLO26 sample in Model Zoo, including algorithm overview, model conversion, runtime inference, model file management, and evaluation notes.

---

## Algorithm Overview

YOLO26 is a real-time vision model series from Ultralytics. This sample provides RDK S100/S100P deployment examples for the following tasks:

- Object Detection
- Instance Segmentation
- Pose Estimation
- Oriented Bounding Box Detection
- Image Classification

- **Official Implementation**: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

### Algorithm Capabilities

- Object detection
- Instance segmentation
- Pose estimation
- Oriented bounding box detection
- Image classification

### Algorithm Features

- **Unified multi-task entry**: `main.py` selects the task through `--task`.
- **Fixed tensor protocol**: each task wrapper parses fixed input and output indexes without guessing output structures.
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
├── evaluator/                      # Accuracy and evaluator scripts
├── model/                          # Model files and download scripts
│   ├── download_model.sh           # Download pre-compiled models
│   └── README.md                   # Model file description
├── runtime/                        # Runtime samples
│   └── python/                     # Python inference sample
│       ├── main.py                 # Python entry script
│       ├── yolo26_det.py           # Detection wrapper
│       ├── yolo26_seg.py           # Segmentation wrapper
│       ├── yolo26_pose.py          # Pose wrapper
│       ├── yolo26_obb.py           # OBB wrapper
│       ├── yolo26_cls.py           # Classification wrapper
│       ├── run.sh                  # One-click execution script
│       └── README.md               # Runtime documentation
├── test_data/                      # Test images and result images
└── README.md                       # Current overview document
```

---

## Quick Start

For a quick experience, run the one-click script under `runtime/python`.

### Python

```bash
cd runtime/python
bash run.sh detect
```

The script downloads the default `yolo26n` detection model if needed and saves the output image into `test_data/`.

For detailed parameters and task examples, refer to [runtime/python/README.md](./runtime/python/README.md).

---

## Model Conversion

This sample provides pre-converted `.hbm` model files for RDK S100/S100P.

- If you only want to run inference, download models from [model/README.md](./model/README.md) and skip conversion.
- If you need to understand or customize conversion, refer to [conversion/README.md](./conversion/README.md).

---

## Runtime

The current sample provides Python runtime implementation.

### Python Version

- Uses `hbm_runtime` as the inference backend
- Provides a unified `Config + Model` wrapper style for all tasks
- Supports zero-argument default execution from `main.py`

For detailed usage, refer to [runtime/python/README.md](./runtime/python/README.md).

---

## Model Evaluation

The `evaluator/` directory is used for task-level accuracy and result export verification. Refer to [evaluator/README.md](./evaluator/README.md) for details.

---

## Inference Result

The Python runtime provides `run.sh` coverage for the following `RDK S100` / `RDK S100P` `.hbm` models:

- `detect`: `n`
- `seg`: `n`
- `pose`: `n`
- `obb`: `n`
- `cls`: `n`

With the default test images, each task should produce boxes, masks, keypoints, oriented boxes, or classification labels that match the image content. Detailed benchmark data and result-check notes are maintained in [evaluator/README.md](./evaluator/README.md).

---

## Inference Results

With the default test images, the task produces visualization results in `test_data/`.

![Detection Result](./test_data/result_detect.jpg)

---

## License

Follows the Model Zoo top-level License.
