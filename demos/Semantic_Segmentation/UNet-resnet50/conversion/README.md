# UNet-resnet50 Model Conversion

[English](./README.md) | [简体中文](./README_cn.md)

## Overview

This directory contains the model conversion toolchain configuration for converting UNet-resnet50 from PyTorch/ONNX to Horizon BPU (.bin) format.

## Directory Structure

```bash
.
|-- ptq_yamls/              # PTQ quantization YAML configurations
|   |-- unet_resnet50_bayese_512x512_nv12.yaml      # Bayes-e (X5) PTQ config
|   `-- unet_resnet50_bernoulli2_512x512_nv12.yaml  # Bernoulli2 (X3) PTQ config
`-- README.md               # This file
```

## Conversion Pipeline

### 1. Environment Preparation

Download the UNet PyTorch implementation:

```bash
git clone https://github.com/bubbliiiing/unet-pytorch.git
cd unet-pytorch
pip install -r requirements.txt
```

### 2. Export to ONNX

Modify `mode` to `export_onnx` in `predict.py`, then run:

```bash
python predict.py
```

### 3. PTQ Quantization and Compilation

Check the model with Horizon OE toolchain:

```bash
hb_mapper checker --model-type onnx --march bayes-e --model UNet_11.onnx
```

Compile the model:

```bash
hb_mapper makertbin --model-type onnx --config unet_resnet50_bernoulli2_512x512_nv12.yaml
```

### 4. Remove Output Dequantize Node (Optional but Recommended)

Check removable dequantize nodes:

```bash
hb_model_modifier unet_resnet50_bayese_512x512_nv12.bin
```

Remove the output dequantize node to improve performance:

```bash
hb_model_modifier unet_resnet50_bayese_512x512_nv12.bin \
    -r "/final/Conv_output_0_HzDequantize"
```

### 5. Verification

Visualize the compiled model:

```bash
hb_perf unet_resnet50_bernoulli2_512x512_nv12.bin
```

Check model I/O information:

```bash
hrt_model_exec model_info --model_file unet_resnet50_bernoulli2_512x512_nv12.bin
```

## Notes

- Pre-converted models are already provided; ordinary users can skip the conversion step.
- Node names may differ during export; please verify carefully using the logs.
- For detailed toolchain documentation, please refer to the Horizon Open Explorer manual.
