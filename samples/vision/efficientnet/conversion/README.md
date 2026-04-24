# Model Conversion

This directory records the conversion-side notes for the EfficientNet sample.

## Overview

The EfficientNet deployment models are provided as RDK X5 `.bin` files. This directory keeps the reference PTQ YAML files used for OE compilation.

If you need to regenerate deployment models, use the OpenExplorer Docker or the corresponding OE package compilation environment.

## Current Assets

This sample keeps the following conversion-related references:

- Published deployment models:
  - `EfficientNet_B2_224x224_nv12.bin`
  - `EfficientNet_B3_224x224_nv12.bin`
  - `EfficientNet_B4_224x224_nv12.bin`
- Runtime input format: packed NV12
- Runtime output: ImageNet-1k classification logits
- Reference PTQ configurations:
  - `EfficientNet_B2_config.yaml`
  - `EfficientNet_B3_config.yaml`
  - `EfficientNet_B4_config.yaml`

The YAML files in this directory are the reference OE/PTQ compilation configurations. They can be used in an OE environment together with `hb_mapper checker` and `hb_mapper makertbin` to regenerate the RDK X5 deployment models.

## ONNX Export Reference

The original demo uses `timm` to export EfficientNet ONNX models. The export flow is:

1. Create the target EfficientNet model with `timm.models.create_model`.
2. Export the model with `torch.onnx.export`.
3. Simplify the ONNX model with `onnxsim.simplify`.
4. Compile the simplified ONNX model in the OE environment.

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
