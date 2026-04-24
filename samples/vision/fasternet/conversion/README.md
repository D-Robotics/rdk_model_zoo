# Model Conversion

This directory records the conversion-side notes for the FasterNet sample.

## Overview

The FasterNet deployment models are provided as RDK X5 `.bin` files. This directory keeps the reference PTQ YAML files used for OE compilation.

If you need to regenerate deployment models, use the OpenExplorer Docker or the corresponding OE package compilation environment.

## Current Assets

This sample keeps the following conversion-related references:

- Published deployment models:
  - `FasterNet_S_224x224_nv12.bin`
  - `FasterNet_T0_224x224_nv12.bin`
  - `FasterNet_T1_224x224_nv12.bin`
  - `FasterNet_T2_224x224_nv12.bin`
- Runtime input format: packed NV12
- Runtime output: ImageNet-1k classification logits
- Reference PTQ configurations:
  - `FasterNet_S_config.yaml`
  - `FasterNet_T0_config.yaml`
  - `FasterNet_T1_config.yaml`
  - `FasterNet_T2_config.yaml`

The YAML files in this directory are the reference OE/PTQ compilation configurations. They can be used in an OE environment together with `hb_mapper checker` and `hb_mapper makertbin` to regenerate the RDK X5 deployment models.

## ONNX Export Reference

The original FasterNet flow uses the official FasterNet source code to export ONNX models. The export flow is:

1. Obtain the official FasterNet source code and pretrained weights from the reference repository.
2. Create the target FasterNet model, such as `fasternet_t0`, `fasternet_t1`, `fasternet_t2`, or `fasternet_s`.
3. Export the model with `torch.onnx.export` using a `1x3x224x224` dummy input.
4. Simplify the ONNX model with `onnxsim.simplify`.
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
