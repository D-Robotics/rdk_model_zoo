English | [简体中文](./README_cn.md)

# SigLIP Model Conversion Guide

This sample uses precompiled SigLIP HBM models. SigLIP vision encoders contain quantization-sensitive structures such as LayerNorm, so the public sample focuses on deployable RDK S100/S100P HBM artifacts.

## Conversion Notes

The SigLIP vision encoder models are quantized and compiled from Google weights published on HuggingFace for Nash BPU deployment. This sample does not provide a reproducible general-purpose conversion script. For deployment, use the `.hbm` models listed in [model/README.md](../model/README.md).

## OE Resources

- OE resource entry point (Docker + OE development package): <https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE toolchain online manual: <https://toolchain.d-robotics.cc/>

## Input Protocol

- Input name: `_input_0`
- Input format: float32 NCHW RGB
- Input range: `[-1, 1]`
- Preprocessing: aspect-ratio resize, RGB `(127, 127, 127)` padding, then `/127.5 - 1.0`

## Submodel Protocol

Each HBM file contains two fixed submodels:

| Submodel | Output Name | Description |
|---|---|---|
| `pooler_output` | `_output_0` | Global image embedding vector |
| `last_hidden_state` | `_output_0` | Patch-level visual features |

## Check Compilation Results

```bash
hrt_model_exec model_info --model_file bpu-siglip-base-patch16-224.hbm
hrt_model_exec perf --thread_num 1 --model_name pooler_output --model_file bpu-siglip-base-patch16-224.hbm
hrt_model_exec perf --thread_num 1 --model_name last_hidden_state --model_file bpu-siglip-base-patch16-224.hbm
```

## License

This directory is licensed under the [Apache 2.0 License](../../../../LICENSE).
