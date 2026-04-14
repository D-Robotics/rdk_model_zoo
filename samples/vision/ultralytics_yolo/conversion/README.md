English | [简体中文](./README_cn.md)

# Ultralytics YOLO Model Conversion Guide

This directory provides tools and instructions for converting Ultralytics YOLO
models to quantized BPU `.bin` models compatible with RDK X5.

## Model Compilation Environment

To convert models, prepare the RDK X5 OpenExplore toolchain on an x86 Linux
machine.

### Method 1: Pip Installation

```bash
conda create -n rdk_env python=3.10 -y
conda activate rdk_env
pip install rdkx5-yolo-mapper
hb_mapper --version
```

### Method 2: Docker Installation

```bash
docker pull openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8
docker run -it --rm -v /path/to/rdk_model_zoo:/data openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
```

## Conversion Workflow

### 1. Export ONNX

Use `export_monkey_patch.py` in an Ultralytics environment to export a `pt`
model to ONNX.

```bash
python3 export_monkey_patch.py --pt yolo11n.pt
```

### 2. Prepare calibration data and mapper config

Use `mapper.py` to prepare calibration data, generate mapper config, and call
`hb_mapper`.

```bash
python3 mapper.py --onnx yolo11n.onnx --cal-images /path/to/calibration_images
```

### 3. Check and compile the BIN model

```bash
hb_mapper checker --model-type onnx --config config.yaml
hb_mapper makertbin --config config.yaml
```

### 4. Inspect the compiled model

```bash
hb_perf config.yaml
hrt_model_exec model_info --model_file yolo11n_detect_bayese_640x640_nv12.bin
hrt_model_exec perf --model_file yolo11n_detect_bayese_640x640_nv12.bin --thread_num 1
```

## Output Tensor Protocol

The Python runtime in this sample uses fixed output parsing. Conversion-side
changes must preserve the following protocols.

### Detection

Supported families:

- `YOLOv5u`
- `YOLOv8`
- `YOLOv9`
- `YOLOv10`
- `YOLO11`
- `YOLO12`
- `YOLO13`

Output order:

- `output[0]`: classification logits on stride `8`
- `output[1]`: DFL box tensor on stride `8`
- `output[2]`: classification logits on stride `16`
- `output[3]`: DFL box tensor on stride `16`
- `output[4]`: classification logits on stride `32`
- `output[5]`: DFL box tensor on stride `32`

### Instance Segmentation

Supported families:

- `YOLOv8`
- `YOLOv9`
- `YOLO11`

Output order:

- `[cls, box, mask_coeff] * 3`
- one final `proto` output

### Pose Estimation

Supported families:

- `YOLOv8`
- `YOLO11`

Output order:

- `[cls, box, keypoints] * 3`

### Classification

Supported families:

- `YOLOv8`
- `YOLO11`

Output order:

- `output[0]`: `(1, 1000, 1, 1)`

## Reference Logs

This directory keeps `hb_mapper` and `hrt_model_exec` reference logs for the
supported model families. These files are used to confirm tensor protocol and
conversion results.
