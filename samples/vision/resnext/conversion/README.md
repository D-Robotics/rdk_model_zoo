# Model Conversion

This directory records the conversion-side notes for the ResNeXt sample.

## Overview

The ResNeXt deployment model is provided as an RDK X5 `.bin` file. This directory keeps the reference PTQ YAML file used for OE compilation.

If you need to regenerate the deployment model, use the OpenExplorer Docker or the corresponding OE package compilation environment.

## Current Assets

This sample keeps the following conversion-related references:

- Published deployment model:
  - `ResNeXt50_32x4d_224x224_nv12.bin`
- Runtime input format: packed NV12
- Runtime output: ImageNet-1k classification logits
- Reference PTQ configuration:
  - `ResNeXt50_32x4d_config.yaml`

The YAML file in this directory is the reference OE/PTQ compilation configuration. It can be used in an OE environment together with `hb_mapper checker` and `hb_mapper makertbin` to regenerate the RDK X5 deployment model.

## ONNX Export Reference

The original ResNeXt flow uses the `timm` model implementation to export ONNX. The export flow is:

1. Install the required Python packages such as `timm`, `onnx`, and `onnxsim`.
2. Create the `resnext50_32x4d` model with pretrained weights.
3. Export the model with `torch.onnx.export` using a `1x3x224x224` dummy input.
4. Simplify the ONNX model with `onnxsim`.
5. Compile the simplified ONNX model in the OE environment.

## Conversion Reference

Please follow the OE package for:

- ONNX preparation
- PTQ configuration generation
- `hb_mapper checker`
- `hb_mapper makertbin`
- `hb_perf`
- `hrt_model_exec`

Offline Docker images can also be obtained from the D-Robotics developer forum: [https://forum.d-robotics.cc/t/topic/28035](https://forum.d-robotics.cc/t/topic/28035).

## Output Protocol

The runtime sample assumes:

- Input tensor shape: `1x3x224x224` before NV12 packing
- Output tensor: ImageNet-1k classification logits
