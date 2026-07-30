# Model Conversion

This directory provides the conversion-side reference for the RepGhost sample.

## Overview

The RepGhost deployment models are provided as RDK X5 `.bin` files. This directory keeps the reference PTQ YAML files used for OE compilation.

If you need to regenerate the deployment models, use the OpenExplorer Docker or the corresponding OE package compilation environment.

## Current Assets

This sample keeps the following conversion-related references:

- Published deployment models:
  - `RepGhost_100_224x224_nv12.bin`
  - `RepGhost_111_224x224_nv12.bin`
  - `RepGhost_130_224x224_nv12.bin`
  - `RepGhost_150_224x224_nv12.bin`
  - `RepGhost_200_224x224_nv12.bin`
- Runtime input format: packed NV12
- Runtime output: ImageNet-1k classification logits
- Reference PTQ configurations:
  - `RepGhost_100.yaml`
  - `RepGhost_111.yaml`
  - `RepGhost_130.yaml`
  - `RepGhost_150.yaml`
  - `RepGhost_200.yaml`

These YAML files are reference OE/PTQ compilation configurations. They can be used in an OE environment together with `hb_mapper checker` and `hb_mapper makertbin` to regenerate the RDK X5 deployment models.

## ONNX Export Reference

The original RepGhost flow exports ONNX from the `timm` implementation. The export reference is:

1. Install the required packages such as `timm`, `onnx`, and `onnxsim`.
2. Create the target RepGhost variant such as `repghostnet_100` with pretrained weights.
3. Export the model with `torch.onnx.export` using a `1x3x224x224` dummy input.
4. Simplify the exported ONNX model with `onnxsim`.
5. Compile the simplified ONNX model in the OE environment.

## Conversion Notes

- The original flow distinguishes multiple variants from `100` to `200`.
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
