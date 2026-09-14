# HGNetV2 Model Conversion and Compilation Guide

English | [简体中文](./README_cn.md)

This directory provides tools and instructions for converting HGNetV2 models into BPU quantized models (`.bin`) compatible with D-Robotics `RDK X5` hardware. Five variants are supported: **b0, b1, b2, b3, b4**.

## Model Compilation Environment

To convert models, you need to install the **RDK X5 OpenExplore Toolchain**.

### Docker Installation

**RDK X5 OpenExplore 1.2.8**
```bash
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker load -i docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
```
Alternatively, obtain the offline Docker image from the D-Robotics Developer Community: [https://forum.d-robotics.cc/t/topic/35229](https://forum.d-robotics.cc/t/topic/35229)

**Start the container** (mount your model zoo so the workspace is shared):
```bash
docker run -it --rm \
  -v /path/to/rdk_model_zoo:/data \
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
```

### Python Dependencies for ONNX Export

The export scripts use the `timm` library to load pretrained PP-HGNetV2 weights. Install it inside the container (or any Python 3 environment with PyTorch ≥ 1.13):

```bash
pip install timm
```

The first run of any `export_hgnetv2_b*_bpu.py` script downloads the pretrained weights from Hugging Face. If your environment has restricted access, set a mirror before launching:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## Conversion Process

### 1. PyTorch (timm) → ONNX

For each variant `${VARIANT}` in `b0 b1 b2 b3 b4`, run:

```bash
cd onnx_export
python3 export_hgnetv2_${VARIANT}_bpu.py
```

This produces `hgnetv2_${VARIANT}.onnx` in `onnx_export/`. The same script is provided for every variant; only the timm model id changes.

### 2. Prepare Calibration Data

`hb_mapper` needs 20–50 representative ImageNet-style images for INT8 calibration. The provided YAML files set `cal_data_dir: '../cal_data'`, so create that folder next to `conversion/`:

```bash
mkdir -p ../cal_data
# Copy 20–50 JPEG images sampled from the ImageNet validation set.
```

### 3. ONNX → BIN

For each variant, run `hb_mapper` from this directory:

```bash
hb_mapper makertbin --model-type onnx --config hgnetv2_${VARIANT}.yaml
```

The resulting `hgnetv2_${VARIANT}_224x224_nv12.bin` is written to `hgnetv2_${VARIANT}_224x224_nv12/`. Move or symlink it into `../model/` so the runtime sample can find it:

```bash
cp hgnetv2_${VARIANT}_224x224_nv12/hgnetv2_${VARIANT}_224x224_nv12.bin ../model/
```

---

## Supported Variants

| Variant | timm model id | Output `.bin` |
| --- | --- | --- |
| b0 | `hgnetv2_b0.ssld_stage2_ft_in1k` | `hgnetv2_b0_224x224_nv12.bin` |
| b1 | `hgnetv2_b1.ssld_stage2_ft_in1k` | `hgnetv2_b1_224x224_nv12.bin` |
| b2 | `hgnetv2_b2.ssld_stage2_ft_in1k` | `hgnetv2_b2_224x224_nv12.bin` |
| b3 | `hgnetv2_b3.ssld_stage2_ft_in1k` | `hgnetv2_b3_224x224_nv12.bin` |
| b4 | `hgnetv2_b4.ssld_stage2_ft_in1k` | `hgnetv2_b4_224x224_nv12.bin` |

---

## License
The tools in this directory follow the [Apache 2.0 License](../../../../LICENSE).
