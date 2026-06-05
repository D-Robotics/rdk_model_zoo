# HGNetv2 Model Conversion and Compilation Guide

English | [简体中文](./README_cn.md)

This directory provides tools and instructions for converting HGNetv2 models into BPU quantized models (`.bin`) compatible with D-Robotics RDK hardware.

## Model Compilation Environment

To convert models, you need to install the **RDK X5 OpenExplore Toolchain**.

### Docker Installation

**RDK X5 OpenExplore 1.2.8**
```bash
docker pull openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8
```
Alternatively, obtain the offline Docker image from the D-Robotics Developer Community: [https://forum.d-robotics.cc/t/topic/28035](https://forum.d-robotics.cc/t/topic/28035)

**Start the container**:
```bash
# Mount your model zoo directory into the container
docker run -it --rm -v /path/to/rdk_model_zoo:/data openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
```

---

## Conversion Process

### 1. pth to onnx Model Conversion

We provide the script `onnx_export/export_hgnetv2_b0_bpu.py` to convert a `.pth` file to an ONNX file.

### 2. onnx to bin Model Conversion

**Prerequisites**:
- An ONNX model adapted for BPU has been exported (refer to `onnx_export/export_hgnetv2_b0_bpu.py`).
- Prepare a folder containing 20–50 images (`.jpg` or `.png`) for quantization calibration.

**Run the conversion**:
```bash
hb_mapper makertbin --model-type onnx --config hgnetv2_b0.yaml
```
After successful conversion, the generated `.bin` model file will be located in the same directory as the ONNX model.

---

## License
The tools in this directory follow the [Apache 2.0 License](../../../../LICENSE).