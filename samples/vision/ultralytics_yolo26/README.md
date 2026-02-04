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
│   ├── download_model.sh  # Script to download Nano models (Fast)
│   └── fulldownload.sh    # Script to download ALL models
├── runtime/        # Inference implementation (Python)
│   └── python/     # Python inference sample (includes run.sh)
├── test_data/      # Directory for saving inference results
└── README.md       # Current model overview documentation
```

## Quick Start
To experience the YOLO26 model quickly, you can run the provided shell script which handles model downloading and execution automatically.

### Python
1. Ensure the requirements are installed on your board.
2. Run the one-click script:
```bash
cd runtime/python
sh run.sh
```
This will download the `yolo26n` detection model (if missing), run inference on a sample image, and save the result to `test_data/result_detect.jpg`.

For more details (parameters, manual execution), please refer to [runtime/python/README.md](./runtime/python/README.md).

## Model Conversion
We provide pre-converted BPU models. If you need to convert your custom models:
1. Export your model to ONNX using the scripts previously located in task folders (now migrating to `conversion/`).
2. Use the toolchain to convert ONNX to `.bin` format.
Detailed instructions can be found in [conversion/README.md](./conversion/README.md).

## Runtime
This sample provides a standardized runtime wrapper for multiple tasks:
- **Python**: Uses `pyeasy_dnn` backend with standardized `Config` and `Model` classes. Each model supports `predict()` and callable interfaces.
- **C++**: (Coming soon).

Detailed runtime documentation:
- [Python Runtime](./runtime/python/README.md)

## Inference Results
After running the sample, the output image will be saved in the `test_data/` directory (e.g., `result_detect.jpg`), showing bounding boxes, masks, or keypoints depending on the selected task.

## Model Evaluation (Evaluator)
The `evaluator/` directory contains scripts for assessing model accuracy, performance, and numerical consistency. You can run these scripts directly on your RDK board to obtain metrics such as mAP and accuracy on standard datasets (e.g., COCO, ImageNet).

For detailed instructions, please refer to: [Model Evaluation README](./evaluator/README.md)

## License
This sample follows the [Apache 2.0 License](../../../LICENSE).
