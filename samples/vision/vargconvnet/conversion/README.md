# Model Conversion

This directory provides the conversion-side notes for the VargConvNet sample.

## Overview

The VargConvNet deployment model is provided as an RDK X5 `.bin` file. This sample does not include a PTQ YAML file or a standalone ONNX export script, so this directory records the deployment protocol and OE reference flow.

If you need to regenerate the deployment model, use the OpenExplorer Docker or the corresponding OE package compilation environment.

## Current Assets

- Published deployment model:
  - `vargconvnet_224x224_nv12.bin`
- Runtime input format: packed NV12
- Runtime output: ImageNet-1k classification logits

## Conversion Reference

Please follow the OE package for:

- ONNX preparation
- PTQ configuration generation
- `hb_mapper checker`
- `hb_mapper makertbin`
- `hb_perf`
- `hrt_model_exec`

Offline Docker images can also be obtained from the D-Robotics developer forum: [https://forum.d-robotics.cc/t/topic/35229](https://forum.d-robotics.cc/t/topic/35229).

## Output Protocol

The runtime sample assumes:

- Input tensor shape: `1x3x224x224` before NV12 packing
- Output tensor: ImageNet-1k classification logits
