English | [简体中文](./README_cn.md)

# PaddleOCR Model Description

This directory describes the complete usage of PaddleOCR in this Model Zoo, including algorithm overview, model conversion, runtime inference (C++ and Python), pre/post-processing interface documentation, and model evaluation.

> Defaults to **PP-OCRv6** (detection + recognition). The run script reads `/sys/class/boardinfo/soc_name` and automatically selects the matching prebuilt model for the current board.

---

## Algorithm Overview

PaddleOCR is an ultra-lightweight Chinese/English OCR system open-sourced by Baidu PaddlePaddle. It uses a two-stage cascaded architecture:

- **Text Detection**: Uses the DB (Differentiable Binarization) algorithm, producing a segmentation probability map of text regions, then extracts polygonal text boxes through thresholding, contour extraction, and minimum-area rectangle fitting.
- **Text Recognition**: Uses the CRNN (Convolutional Recurrent Neural Network) architecture, outputting per-timestep class probabilities (logits), decoded into text strings via CTC greedy decoding.

- **Papers**:
  - DB: [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947)
  - CRNN: [An End-to-End Trainable Neural Network for Image-based Sequence Recognition](https://arxiv.org/abs/1507.05717)
- **Reference Implementation**: [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

### Algorithm Capabilities

- Chinese / English text detection and recognition on a single image
- Polygon text bounding boxes plus per-line recognised text
- Supports mixed-language scenes with the PP-OCRv6 default dictionary

### Algorithm Features

- **Two-stage decoupled**: detection and recognition run independently, making it easy to swap each stage's model
- **DB text detection**: differentiable binarisation lets the model learn the binarisation threshold directly during training, improving small-text detection accuracy
- **CRNN + CTC recognition**: sequence modelling combined with CTC decoding, natively supporting variable-length text
- **Chinese support**: dictionary covers common Chinese characters and symbols, supporting mixed Chinese/English scenarios
- **Edge-optimised**: PP-OCRv6 has been quantised and compiled for RDK S100 / S600 BPU

---

## Platform Compatibility

This sample defaults to **PP-OCRv6** detection/recognition models, automatically selecting the matching prebuilt variant based on `/sys/class/boardinfo/soc_name`:

| Platform | Supported | Model variant |
|---|---|---|
| RDK S100 | ✅ Supported | `archive.d-robotics.cc/.../rdk_s100/paddle_ocr/` |
| RDK S100P | ✅ Supported | Reuses `rdk_s100/paddle_ocr/` prebuilt models |
| RDK S600 | ✅ Supported | `archive.d-robotics.cc/.../rdk_s600/paddle_ocr/` |

> When `soc_name` read fails or returns `(null)`, the script falls back to S100 models.
> To run OCR on a custom platform, re-quantise and compile using the matching OE toolchain; see [conversion/README.md](./conversion/README.md).

---

## Directory Structure

```bash
.
|-- conversion                          # Model conversion workflow
|   |-- paddleocr_det_configs.yaml      # PP-OCRv6 detection quantisation config
|   |-- paddleocr_rec_configs.yaml      # PP-OCRv6 recognition quantisation config
|   `-- README.md                       # Conversion guide
|-- evaluator                           # Model evaluation
|   `-- README.md                       # Evaluation guide
|-- model                               # Model artifacts and download script
|   |-- download_model.sh               # SOC-aware HBM download script
|   `-- README.md                       # Model download guide
|-- runtime                             # Inference samples
|   |-- cpp                             # C++ inference project
|   |   |-- inc                         # C++ headers
|   |   |   `-- paddle_ocr.hpp          # PaddleOCR C++ wrapper interface
|   |   |-- src                         # C++ source
|   |   |   |-- main.cpp                # Inference entry
|   |   |   `-- paddle_ocr.cpp          # PaddleOCR inference implementation
|   |   |-- CMakeLists.txt              # CMake build configuration
|   |   |-- README.md                   # C++ inference guide
|   |   `-- run.sh                      # C++ run script
|   `-- python                          # Python inference sample
|       |-- README.md                   # Python inference guide
|       |-- main.py                     # Python inference entry
|       |-- run.sh                      # Python run script
|       `-- paddle_ocr.py               # PaddleOCR inference & post-processing
|-- test_data                           # Test data
|   |-- gt_2322.jpg                     # Sample test image (with Chinese text)
|   |-- ppocrv6_dict.txt                # PP-OCRv6 default character dictionary
|   |-- ppocrv6_tiny_dict.txt           # PP-OCRv6 Tiny fallback dictionary
|   |-- ppocr_keys_v1.txt               # PP-OCRv3 legacy dictionary (compat)
|   `-- FangSong.ttf                    # FangSong font file (for result visualisation)
`-- README.md                           # This file — overall guide
```

---

## Quick Start

Each model provides a `run.sh` script for one-click execution:

- Detects system environment and installs missing dependencies automatically
- Reads the board SOC and auto-selects PP-OCRv6 model for S100 or S600
- Checks for HBM model files and downloads them if missing
- Creates the `build` directory and compiles the C++ project (C++ only)
- Runs the compiled binary or Python script

### C++

```bash
cd runtime/cpp/
./run.sh
```

For detailed step-by-step guidance, see `runtime/cpp/README.md`.

### Python

```bash
cd runtime/python/
./run.sh
```

For detailed step-by-step guidance, see `runtime/python/README.md`.

---

## Model Conversion

- Pre-built PP-OCRv6 HBM models (S100 / S600) are available through `model/download_model.sh`. If you do not need to customise conversion parameters, **skip this section**.
- For custom conversion parameters or the full conversion workflow, see `conversion/README.md`.

---

## Runtime

PaddleOCR provides both C++ and Python inference samples. Both produce identical results.

### C++ Version

- Full engineering project, suitable for production integration
- Includes model wrapper, argument parsing, inference pipeline, and build instructions
- Default model path is resolved at compile time based on SOC (S100/S600), overridable via `--det_model_path` / `--rec_model_path`
- See `runtime/cpp/README.md` for details

### Python Version

- Script-based, suitable for quick prototyping and algorithm validation
- Default model path is resolved at runtime based on SOC (S100/S600), overridable via `--det-model-path` / `--rec-model-path`
- Demonstrates model loading, inference, post-processing, and result saving
- See `runtime/python/README.md` for details

---

## Model Evaluation

The `evaluator/` directory provides model accuracy, performance, and numerical consistency checks. See the README there for details.

---

## Inference Result

After inference, `result.jpg` is generated in the working directory:

- **Left half**: original image with detected text boxes overlaid (green polygon outlines), each box marks a detected text region
- **Right half**: white canvas with recognised text rendered in FangSong font, each line's vertical position aligned with the corresponding detection box centre

C++ inference result example:

![C++ result](test_data/cpp_demo.jpg)

Python inference result example:

![Python result](test_data/python_demo.jpg)

---

## License

Follows the top-level Model Zoo License.