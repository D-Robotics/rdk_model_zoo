[English](./README.md) | [简体中文](./README_cn.md)

# YOLO26 Depth

This sample provides model conversion, evaluation utilities, board performance data, and Python/C++ inference for Ultralytics YOLO26 monocular depth models on RDK X5.

## Algorithm Overview

YOLO26 Depth predicts dense relative depth from one RGB image. The compiled BPU model outputs calibrated log-depth with shape `1x192x192x1`. Exponential decoding, bilinear resizing, and letterbox restoration run on the CPU.

Reference: [Ultralytics Depth](https://docs.ultralytics.com/tasks/depth/)

## Capabilities

- Supports YOLO26n, YOLO26s, YOLO26m, YOLO26l, and YOLO26x Depth.
- Provides a complete PyTorch weights → ONNX → PTQ → RDK X5 BIN conversion path.
- Provides Python and C++ runtime implementations with consistent preprocessing and postprocessing.
- Provides single-image numerical comparison and SUN RGB-D subset evaluation tools.

## Platform Compatibility

| Platform | Runtime model | Python | C++ |
| --- | --- | --- | --- |
| RDK X5 | `.bin` | Supported | Supported |

The runtime was validated with RDK X5 BSP 3.5.0-beta and DNN Runtime 1.24.5. Python inference must use the BSP-provided `hbm_runtime` package matching the installed `libdnn`.

## Directory Structure

```text
yolo26_depth/
├── conversion/           # ONNX export, calibration preparation, and PTQ scripts
├── evaluator/            # Numerical and SUN RGB-D evaluation tools
├── model/                # Model download script and model metadata
├── runtime/
│   ├── cpp/              # C++ inference implementation
│   └── python/           # Python inference implementation
├── test_data/            # Example input image
├── README.md
└── README_cn.md
```

## Quick Start

`model/download_model.sh` uses the official archive URL. Download the default model with:

```bash
bash model/download_model.sh n
```

Run Python inference with default arguments:

```bash
cd runtime/python
bash run.sh
```

Run C++ inference with default arguments:

```bash
cd runtime/cpp
bash run.sh
```

Both scripts default to the YOLO26n 768 model and `test_data/bus.jpg`. A different model, input image, and output directory can be provided as positional arguments:

```bash
bash run.sh MODEL.bin INPUT.jpg OUTPUT_DIR
```

## Model Conversion

See [conversion/README.md](conversion/README.md) for ONNX export, calibration-data preparation, PTQ configuration, and Mapper execution.

## Model Inference

- [Python runtime](runtime/python/README.md)
- [C++ runtime](runtime/cpp/README.md)

The Python runtime writes `log_depth.npy`, `depth_native.npy`, `depth.png`, `overlay.png`, and `report.json`. The C++ runtime writes `depth_native.f32`, `depth.png`, `overlay.png`, and `report.json`.

## Model Evaluation And Performance

See [evaluator/README.md](evaluator/README.md) for RDK X5 board performance data, single-image numerical comparison, and SUN RGB-D subset evaluation. Board accuracy data is not published for this sample yet; use the evaluator tools to generate it when validated outputs are available.

## Source Reference

Follow the [source-reference documentation guide](../../../docs/source_reference/README.md) to generate and browse API documentation.

## Notes

- The output is relative depth rather than calibrated metric depth.
- The test image is intended for functional verification, not accuracy benchmarking.
- Generated models, datasets, logs, and evaluation outputs must not be committed to the sample directory.

## License

The sample code follows the repository license. Ultralytics models and SUN RGB-D data remain subject to their respective upstream licenses.
