# Model Conversion

This directory provides the conversion-side reference for the MobileOne sample.

## Overview

The MobileOne deployment models are provided as RDK X5 `.bin` files. This directory keeps the reference PTQ YAML files used for OE compilation.

If you need to regenerate the deployment models, use the OpenExplorer Docker or the corresponding OE package compilation environment.

## Current Assets

This sample keeps the following conversion-related references:

- Published deployment models:
  - `MobileOne_S0_224x224_nv12.bin`
  - `MobileOne_S1_224x224_nv12.bin`
  - `MobileOne_S2_224x224_nv12.bin`
  - `MobileOne_S3_224x224_nv12.bin`
  - `MobileOne_S4_224x224_nv12.bin`
- Runtime input format: packed NV12
- Runtime output: ImageNet-1k classification logits
- Reference PTQ configurations:
  - `MobileOne_S0_config.yaml`
  - `MobileOne_S1_config.yaml`
  - `MobileOne_S2_config.yaml`
  - `MobileOne_S3_config.yaml`
  - `MobileOne_S4_config.yaml`

These YAML files are reference OE/PTQ compilation configurations. They can be used in an OE environment together with `hb_mapper checker` and `hb_mapper makertbin` to regenerate the RDK X5 deployment models.

## ONNX Export Reference

The original MobileOne flow exports ONNX from the official source repository. The export reference is:

1. Install the required packages such as `torch`, `onnx`, and `onnxsim`.
2. Clone the official `apple/ml-mobileone` repository.
3. Load the unfused checkpoint for the target variant such as `mobileone_s0_unfused.pth.tar`.
4. Create the corresponding MobileOne model and run `reparameterize_model(model)`.
5. Export the reparameterized model with `torch.onnx.export` using a `1x3x224x224` dummy input.
6. Simplify the exported ONNX model with `onnxsim`.
7. Compile the simplified ONNX model in the OE environment.

## Conversion Notes

- The original flow distinguishes multiple variants from `S0` to `S4`.
- The YAML files in this directory correspond to those published variants.
- Select the matching YAML file when regenerating the target `.bin` model.

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
