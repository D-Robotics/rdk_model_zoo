English | [简体中文](./README_cn.md)

# MobileNetV2

This directory describes the complete usage of MobileNetV2 in this Model Zoo, including algorithm overview, model conversion, runtime inference (Python / C++), reusable pre/post-processing interfaces, and model evaluation.

---

## Algorithm Overview

MobileNetV2 is a lightweight convolutional neural network designed for mobile and embedded image classification. It features:

- **Inverted Residuals** with Linear Bottlenecks: reduces computation while maintaining accuracy
- **Depthwise Separable Convolutions**: splits standard convolution into depthwise + pointwise stages, significantly reducing parameters and FLOPs
- **Lightweight and efficient**: far fewer parameters than ResNet families, making it suitable for resource-constrained platforms
- **Strong generalisation**: pre-trained on ImageNet, directly applicable to 1000-class image classification

### Capabilities

- Multi-class classification of a single image
- Confidence scores and Top-K predictions for each class

### References

- MobileNetV2 paper: [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- PyTorch official implementation: https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py

---

## Directory Structure

```bash
.
|-- conversion                          # Model conversion workflow
|   `-- README.md                       # Conversion guide
|-- evaluator                           # Model evaluation
|   `-- README.md                       # Evaluation guide
|-- model                               # Model artifacts and download script
|   `-- download_model.sh               # HBM model download script
|-- runtime                             # Inference samples
|   |-- cpp                             # C++ inference project
|   |   |-- inc                         # C++ headers
|   |   |   `-- mobilenetv2.hpp         # MobileNetV2 C++ wrapper interface
|   |   |-- src                         # C++ source
|   |   |   |-- main.cpp                # Inference entry
|   |   |   `-- mobilenetv2.cpp         # MobileNetV2 inference implementation
|   |   |-- CMakeLists.txt              # CMake build configuration
|   |   |-- README.md                   # C++ inference guide
|   |   `-- run.sh                      # C++ run script
|   `-- python                          # Python inference sample
|       |-- README.md                   # Python inference guide
|       |-- main.py                     # Python inference entry
|       |-- run.sh                      # Python run script
|       `-- mobilenetv2.py              # MobileNetV2 inference & post-processing
|-- test_data                           # Test data
|   |-- zebra_cls.jpg                   # Sample test image
|   `-- imagenet1000_labels.txt         # ImageNet 1000-class label file
`-- README.md                           # This file — overall guide
```

---

## QuickStart

Each model provides a `run.sh` script for one-click execution:

- Detects the system environment and installs missing dependencies automatically
- Detects the board SOC (S100 / S600) and downloads the matching HBM model
- Creates the `build` directory and compiles the C++ project (C++ only)
- Runs the compiled binary or Python script

### C++

```bash
cd runtime/cpp/
./run.sh
```

For detailed step-by-step guidance refer to `runtime/cpp/README.md`.

### Python

```bash
cd runtime/python/
./run.sh
```

For detailed step-by-step guidance refer to `runtime/python/README.md`.

---

## Model Conversion

- Pre-built HBM models are available through the `model/download_model.sh` script. If you do not need to customise conversion parameters, **skip this section**.
- For custom conversion parameters or the full conversion workflow, see `conversion/README.md`.

---

## Runtime

MobileNetV2 provides both C++ and Python inference samples. Both produce identical results.

### C++ Version

- Full engineering project, suitable for production integration
- Includes model wrapper, argument parsing, inference pipeline, and build instructions
- See `runtime/cpp/README.md` for details

### Python Version

- Script-based, suitable for quick prototyping and algorithm validation
- Demonstrates model loading, inference, post-processing, and result visualisation
- See `runtime/python/README.md` for details

---

## Evaluation

The `evaluator/` directory provides model accuracy, performance, and numerical consistency checks. See the README there for details.

---

## Inference Result

With `zebra_cls.jpg` as input, the Top-5 results are:

```bash
Top-5 Classification Results:
  [0] zebra: 0.9922
  [1] tiger, Panthera tigris: 0.0040
  [2] hartebeest: 0.0013
  [3] tiger cat: 0.0007
  [4] impala, Aepyceros melampus: 0.0005
```

---

## License

Follows the top-level Model Zoo License.