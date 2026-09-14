English | [简体中文](./README_cn.md)

# YOLO26 Model Description

This directory provides the complete usage guide for the YOLO26 sample in Model Zoo, including algorithm overview, model conversion, runtime inference, model file management, and evaluation notes.

---

## Algorithm Overview

YOLO26 is a real-time vision model series from Ultralytics. This sample provides RDK X5 deployment examples for the following tasks:

- Object Detection
- Instance Segmentation
- Pose Estimation
- Oriented Bounding Box Detection
- Image Classification

- **Official Implementation**: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

### Platform Notes

- Target platform: `RDK X5`
- Runtime backend: `hbm_runtime`
- Inference model format: `.bin`
- Input format: `NV12`

---

## Directory Structure

```bash
.
├── conversion/                     # Model conversion workflow
├── evaluator/                      # Accuracy and evaluator scripts
├── model/                          # Model files and download scripts
│   ├── download_model.sh           # Download nano models
│   ├── fulldownload.sh             # Download all models
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
├── test_data/                      # Inference results
└── README.md                       # Current overview document
```

---

## QuickStart

For a quick experience, run the one-click script under `runtime/python`.

### Python

```bash
cd runtime/python
chmod +x run.sh
./run.sh
```

The script downloads the default `yolo26n` detection model if needed and saves the output image into `test_data/`.

For detailed parameters and task examples, refer to [runtime/python/README.md](./runtime/python/README.md).

---

## Model Conversion

This sample provides pre-converted `.bin` model files for RDK X5.

- If you only want to run inference, download models from [model/README.md](./model/README.md) and skip conversion.
- If you need to understand or customize conversion, refer to [conversion/README.md](./conversion/README.md).

---

## Runtime Inference

The current sample provides Python runtime implementation.

### Python Version

- Uses `hbm_runtime` as the inference backend
- Provides a unified `Config + Model` wrapper style for all tasks
- Supports zero-argument default execution from `main.py`

For detailed usage, refer to [runtime/python/README.md](./runtime/python/README.md).

---

## Evaluator

The `evaluator/` directory is used for task-level accuracy and result export verification. Refer to [evaluator/README.md](./evaluator/README.md) for details.

---

## Validation Status

The current Python sample has been verified on `RDK X5` with the following `.bin` models:

- `detect`: `n / s / m / l / x`
- `seg`: `n / s / m / l / x`
- `pose`: `n / s / m / l / x`
- `obb`: `n / s / m / l / x`
- `cls`: `n / s / m / l / x`

---

## License

Follows the Model Zoo top-level License.
