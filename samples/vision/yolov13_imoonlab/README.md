English | [简体中文](./README_cn.md)

# YOLOv13 iMoonLab Model Description

This directory provides the full YOLOv13 iMoonLab detection sample documentation for Model Zoo, including the algorithm overview, model conversion, runtime usage, model file management, and evaluation notes.

## Algorithm Overview

YOLOv13 is a real-time object detector released by the Intelligent Media and Cognition Laboratory of Tsinghua University. Its design combines HyperACE, FullPAD, and lightweight convolution replacements to balance accuracy, speed, and parameter efficiency on generic object detection tasks such as COCO.

- Official repository: <https://github.com/iMoonLab/yolov13>
- Paper: <https://arxiv.org/abs/2506.17733>
- Project page: <https://www.gaoyue.org/>

![YOLOv13 icon](test_data/icon.png)

![YOLOv13 framework](test_data/framework.png)

## Algorithm Capabilities

- Object detection

## Algorithm Features

- Focused YOLOv13 Detect sample with a single clear runtime path.
- Python runtime follows the `Config + Wrapper + predict()` structure.
- HBM model input is fixed to dual-input NV12: Y plane + UV plane.
- Postprocess parses tensors by fixed output indexes instead of guessing layouts.

## Directory Structure

```bash
.
├── conversion/              # Export, calibration, and compile notes
├── evaluator/               # Accuracy and performance evaluation notes
├── model/                   # Model download script and model notes
├── runtime/
│   └── python/              # Python runtime sample
├── test_data/               # Test assets, labels, and documentation images
├── README.md
└── README_cn.md
```

## Quick Start

Run the default board-side demo from `runtime/python/`:

```bash
cd runtime/python
bash run.sh
```

The script checks and downloads `../../model/s100/yolo13n_detect_nashe_640x640_nv12.hbm` when needed, then runs detection on `../../test_data/kite.jpg`.

For manual model and image selection, see [runtime/python/README.md](./runtime/python/README.md).

## Model Conversion

ONNX export, calibration data preparation, and HBM compile instructions are documented in [conversion/README.md](./conversion/README.md).

## Runtime

The Python runtime uses `hbm_runtime`. Runtime arguments and direct `python3 main.py` examples are documented in [runtime/python/README.md](./runtime/python/README.md).

## Model Evaluation

Performance tables, accuracy tables, and evaluation methodology are documented in [evaluator/README.md](./evaluator/README.md).

## Performance Data

This sample covers the following reference models:

- `yolo13n_detect_nashe_640x640_nv12.hbm`
- `yolo13s_detect_nashe_640x640_nv12.hbm`
- `yolo13l_detect_nashe_640x640_nv12.hbm`
- `yolo13x_detect_nashe_640x640_nv12.hbm`

The corresponding performance and accuracy data are maintained in [evaluator/README.md](./evaluator/README.md).

## License

This sample follows the repository top-level `LICENSE`.
