English | [简体中文](./README_cn.md)

# Ultralytics YOLO Model Description

This directory describes the Ultralytics YOLO sample in the Model Zoo,
including algorithm overview, model conversion, runtime inference, model file
management, and evaluator usage.

---

## Algorithm Overview

Ultralytics YOLO is a real-time vision model family covering object detection,
instance segmentation, pose estimation, and image classification. This sample
provides RDK X5 deployment examples for the following model families:

- Detection:
  `YOLOv5u / YOLOv8 / YOLOv9 / YOLOv10 / YOLO11 / YOLO12 / YOLO13`
- Instance Segmentation:
  `YOLOv8 / YOLOv9 / YOLO11`
- Pose Estimation:
  `YOLOv8 / YOLO11`
- Image Classification:
  `YOLOv8 / YOLO11`

- **Official Implementation**: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

### Platform Notes

- Target platform: `RDK X5`
- Runtime backend: `hbm_runtime`
- Inference model format: `.bin`
- Input format: packed `NV12`

---

## Directory Structure

```bash
.
|-- conversion/                     # Model conversion workflow
|-- evaluator/                      # Accuracy and benchmark documents
|-- model/                          # Model files and download scripts
|   |-- download_model.sh           # Download default models
|   |-- fulldownload.sh             # Download all supported models
|   |-- README.md                   # Model file description
|   `-- README_cn.md                # 中文模型文件说明
|-- runtime/                        # Runtime samples
|   |-- cpp/                        # C++ reference runtime
|   `-- python/                     # Python runtime sample
|       |-- main.py                 # Python entry script
|       |-- ultralytics_yolo_det.py # Detection wrapper
|       |-- ultralytics_yolo_seg.py # Segmentation wrapper
|       |-- ultralytics_yolo_pose.py# Pose wrapper
|       |-- ultralytics_yolo_cls.py # Classification wrapper
|       |-- run.sh                  # One-click execution script
|       |-- README.md               # Python runtime documentation
|       `-- README_cn.md            # Python runtime 中文文档
|-- test_data/                      # Test images and result images
|-- README.md                       # Overview document
`-- README_cn.md                    # 中文总览文档
```

---

## QuickStart

For a quick experience, run the script under `runtime/python`.

### Python

```bash
cd runtime/python
chmod +x run.sh
./run.sh
```

The default command downloads `yolo11n_detect_bayese_640x640_nv12.bin` if
needed and saves the result image into `test_data/`.

For detailed arguments and task examples, refer to
[runtime/python/README.md](./runtime/python/README.md).

---

## Model Conversion

This sample provides pre-converted `.bin` models for RDK X5.

- If you only want to run inference, download models from [model/README.md](./model/README.md) and skip conversion.
- If you need to export ONNX, prepare calibration data, or compile the model,
  refer to [conversion/README.md](./conversion/README.md).

---

## Runtime Inference

This sample provides Python and C++ runtime directories.

### Python Version

- Uses `hbm_runtime` as the inference backend
- Uses one unified `main.py` entry for all tasks
- Uses `Config + Wrapper + predict()` for each task

For detailed usage, refer to
[runtime/python/README.md](./runtime/python/README.md).

### C++ Version

The C++ directory is kept for reference. Refer to
[runtime/cpp/README.md](./runtime/cpp/README.md).

---

## Evaluator

The `evaluator/` directory contains benchmark tables, accuracy references, and
runtime validation records for the supported models.

Refer to [evaluator/README.md](./evaluator/README.md) for details.

---

## Validation Status

The Python sample has been verified on `RDK X5` for all documented models in
this directory:

- Detect:
  `YOLOv5u / YOLOv8 / YOLOv9 / YOLOv10 / YOLO11 / YOLO12 / YOLO13`
- Seg:
  `YOLOv8 / YOLOv9 / YOLO11`
- Pose:
  `YOLOv8 / YOLO11`
- CLS:
  `YOLOv8 / YOLO11`

Detailed benchmark data and validation summaries are maintained in
[evaluator/README.md](./evaluator/README.md).

---

## License

Follows the Model Zoo top-level License.
