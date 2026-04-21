# Model Conversion

This directory records the conversion-side notes for the MobileNetV3 sample.

## Overview

The complete MobileNetV3 conversion workflow is not maintained in this repository.  
If you need to regenerate the deployment model, please refer to the OE package for the complete conversion process.

## Current Assets

This sample keeps the following conversion-related references:

- Published deployment model name: `MobileNetV3_224x224_nv12.bin`
- Runtime input format: packed NV12
- Runtime output: ImageNet-1k classification logits
- Reference PTQ configuration: `MobileNetV3_config.yaml`

## Conversion Reference

Please follow the OE package for:

- ONNX export
- PTQ configuration generation
- `hb_mapper checker`
- `hb_mapper makertbin`
- `hb_perf`
- `hrt_model_exec`

## Output Protocol

The runtime sample assumes:

- Input tensor shape: `1x3x224x224` before NV12 packing
- Output tensor: ImageNet-1k classification logits
