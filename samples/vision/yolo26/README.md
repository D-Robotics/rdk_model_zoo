# YOLO26 Model Sample

English | [简体中文](./README_cn.md)

## Algorithm Overview
YOLO26 is a versatile and high-performance real-time model series. This sample provides deployment routines for multiple tasks including Detection, Instance Segmentation, Pose Estimation, Oriented Bounding Box (OBB), and Image Classification on D-Robotics RDK hardware.

For more information, please refer to the official [Ultralytics](https://github.com/ultralytics/ultralytics) resources.

## Directory Structure
```bash
.
├── conversion/     # Model conversion workflow (ONNX -> BIN)
├── evaluator/      # Accuracy and performance evaluation
├── model/          # Model files and download scripts
├── runtime/        # Inference implementation (Python / C++)
│   └── python/     # Python inference sample
├── test_data/      # Sample input images for testing
└── README.md       # Current model overview documentation
```

## Quick Start
To experience the YOLO26 model quickly, you can run the inference script on your RDK board.

### Python
1. Ensure the requirements are installed on your board.
2. Run the unified inference entry:
```bash
cd runtime/python
python3 main.py --task detect --model-path ../../model/yolo26n_bpu_bayese_640x640_nv12.bin --test-img ../../test_data/bus.jpg
```
For more details (parameters, environment setup), please refer to [runtime/python/README.md](./runtime/python/README.md).

## Model Conversion
We provide pre-converted BPU models. If you need to convert your custom models:
1. Export your model to ONNX using the scripts previously located in task folders (now migrating to `conversion/`).
2. Use the toolchain to convert ONNX to `.bin` format.
Detailed instructions can be found in [conversion/README.md](./conversion/README.md).

## Runtime
This sample provides a standardized runtime wrapper for multiple tasks:
- **Python**: Uses `pyeasy_dnn` backend (standardized with Config and Model classes).
- **C++**: (Coming soon).

Detailed runtime documentation:
- [Python Runtime](./runtime/python/README.md)

## Inference Results
After running the sample, the output image will be saved (e.g., `result.jpg`), showing bounding boxes, masks, or keypoints depending on the selected task.

## Model Evaluation (Evaluator)
The `evaluator/` directory contains scripts for assessing model accuracy, performance, and numerical consistency. You can run these scripts directly on your RDK board to obtain metrics such as mAP and accuracy on standard datasets (e.g., COCO, ImageNet).

For detailed instructions, please refer to: [Model Evaluation README](./evaluator/README.md)

## License
This sample follows the [Apache 2.0 License](../../../LICENSE).