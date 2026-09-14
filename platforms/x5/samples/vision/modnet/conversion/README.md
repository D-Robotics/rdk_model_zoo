# Model Conversion

This directory provides the conversion assets for building RDK X5 deployment models for the MODNet portrait matting sample.

The current sample runtime uses `.bin` models with `hbm_runtime`. If you only need to run inference, use the prebuilt model in [`../model/README.md`](../model/README.md). This document is only for rebuilding the deployment model from an ONNX source model.

## Directory Structure

```text
.
├── onnx_export                     # ONNX export scripts
├── ptq_yamls                       # PTQ configuration YAML files
├── README.md
└── README_cn.md
```

## Prerequisites

Prepare the following before conversion:

1. Install the RDK X5 OpenExplorer toolchain with `hb_mapper`, `hb_perf`, and `hrt_model_exec`.
2. Prepare the MODNet ONNX model.
3. Prepare calibration data for PTQ.

## Prepare ONNX

The original MODNet model is from the official implementation:

- Paper: [Is a Green Screen Really Necessary for Real-Time Portrait Matting?](https://arxiv.org/abs/2011.11961)
- Repository: [ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet)

Prepare the ONNX model first. Use the official MODNet project or your own export pipeline to generate the ONNX model, then update the `onnx_model` field in the selected YAML file before running `hb_mapper`.

## PTQ Conversion

Run `hb_mapper checker` first to verify the ONNX model:

```bash
hb_mapper checker --config modnet.yaml
```

Then build the deployment model:

```bash
hb_mapper makertbin --config modnet.yaml
```

## Validation Commands

Use `hb_perf` to inspect performance after conversion:

```bash
hb_perf model_perf \
    --model ./modnet_512x512_rgb.bin \
    --input-shape input 1x3x512x512
```

Use `hrt_model_exec` for basic runtime verification:

```bash
hrt_model_exec perf \
    --model_file ./modnet_512x512_rgb.bin \
    --thread_num 1
```

## Runtime Protocol

The generated deployment model is expected to follow this runtime protocol:

- Input tensor type: `Float32 NCHW RGB`
- Input resolution: `512x512`
- Input normalization: `(pixel - 127.5) / 127.5` (range [-1, 1])
- Output tensor shape: `1x1x512x512`
- Output tensor type: `F32` (alpha matte in [0, 1])

This protocol matches the Python runtime under [`../runtime/python/README.md`](../runtime/python/README.md).
