# Model Conversion

This directory provides the conversion-side reference for the RepVGG sample.

## Overview

The RepVGG deployment models are provided as RDK X5 `.bin` files. This directory keeps the reference PTQ YAML files used for OE compilation.

If you need to regenerate the deployment models, use the OpenExplorer Docker or the corresponding OE package compilation environment.

## Current Assets

This sample keeps the following conversion-related references:

- Published deployment models:
  - `RepVGG_A0_224x224_nv12.bin`
  - `RepVGG_A1_224x224_nv12.bin`
  - `RepVGG_A2_224x224_nv12.bin`
  - `RepVGG_B0_224x224_nv12.bin`
  - `RepVGG_B1g2_224x224_nv12.bin`
  - `RepVGG_B1g4_224x224_nv12.bin`
- Runtime input format: packed NV12
- Runtime output: ImageNet-1k classification logits
- Reference PTQ configurations:
  - `RepVGG_A0_config.yaml`
  - `RepVGG_A1_config.yaml`
  - `RepVGG_A2_config.yaml`
  - `RepVGG_B0_config.yaml`
  - `RepVGG_B1g2_config.yaml`
  - `RepVGG_B1g4_config.yaml`

These YAML files are reference OE/PTQ compilation configurations. They can be used in an OE environment together with `hb_mapper checker` and `hb_mapper makertbin` to regenerate the RDK X5 deployment models.

## ONNX Export Reference

The original RepVGG flow exports ONNX from the official RepVGG implementation. The export reference is:

1. Prepare the official RepVGG source code and the corresponding pretrained weights.
2. Create the target model variant, such as `create_RepVGG_B1g2(deploy=False)`.
3. Load the training checkpoint and convert it with `repvgg_model_convert()`.
4. Export the converted model with `torch.onnx.export` using a `1x3x224x224` dummy input.
5. Compile the exported ONNX model in the OE environment.

## Conversion Notes

- The original flow distinguishes `A0`, `A1`, `A2`, `B0`, `B1g2`, and `B1g4` variants.
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
