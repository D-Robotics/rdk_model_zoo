# Model Conversion

This directory provides the conversion assets for building RDK X5 deployment models for the ConvNeXt image classification sample.

The current sample runtime uses `.bin` models with `hbm_runtime`. If you only need to run inference, use the prebuilt model in [`../model/README.md`](../model/README.md). This document is only for rebuilding the deployment model from an ONNX source model.

## Directory Structure

```text
.
├── ConvNeXt_atto.yaml     # PTQ configuration for ConvNeXt Atto
├── ConvNeXt_femto.yaml    # PTQ configuration for ConvNeXt Femto
├── ConvNeXt_nano.yaml     # PTQ configuration for ConvNeXt Nano
├── README.md
└── README_cn.md
```

## Supported Variants

The current conversion assets cover the following ConvNeXt variants:

| YAML File | Variant | Runtime Input | Target Platform |
| --- | --- | --- | --- |
| `ConvNeXt_atto.yaml` | ConvNeXt Atto | `224x224 NV12` | `RDK X5 / bayes-e` |
| `ConvNeXt_femto.yaml` | ConvNeXt Femto | `224x224 NV12` | `RDK X5 / bayes-e` |
| `ConvNeXt_nano.yaml` | ConvNeXt Nano | `224x224 NV12` | `RDK X5 / bayes-e` |

## Prerequisites

Prepare the following before conversion:

1. Install the RDK X5 OpenExplorer toolchain with `hb_mapper`, `hb_perf`, and `hrt_model_exec`.
2. Prepare the ConvNeXt ONNX model that matches the target variant.
3. Prepare calibration data for PTQ. The YAML files expect RGB float calibration data under `./calibration_data_rgb_f32`.

## Prepare ONNX

The original ConvNeXt model is from the official implementation:

- Paper: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- Repository: [facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)

Prepare the ONNX model for the target variant first. This sample does not provide an ONNX export script in the repository. Use the official ConvNeXt project or your own export pipeline to generate the matching ONNX model, then update the `onnx_model` field in the selected YAML file before running `hb_mapper`.

## PTQ Conversion

Run `hb_mapper checker` first to verify the ONNX model:

```bash
hb_mapper checker --config ConvNeXt_atto.yaml
```

Then build the deployment model:

```bash
hb_mapper makertbin --config ConvNeXt_atto.yaml
```

Repeat the same workflow for `ConvNeXt_femto.yaml` or `ConvNeXt_nano.yaml` when converting other variants.

## YAML Notes

The YAML files in this directory share the same deployment protocol:

- `march: bayes-e`
- runtime input type: `nv12`
- training input type: `rgb`
- training layout: `NCHW`
- normalization: `data_mean_and_scale`
- output model prefix: `ConvNeXt-deploy_224x224_nv12`

Before conversion, confirm these fields in the selected YAML:

- `onnx_model`
- `cal_data_dir`
- `working_dir`
- `output_model_file_prefix`

## Validation Commands

Use `hb_perf` to inspect performance after conversion:

```bash
hb_perf model_perf \
    --model ./ConvNeXt-deploy_224x224_nv12.bin \
    --input-shape data 1x3x224x224
```

Use `hrt_model_exec` for basic runtime verification:

```bash
hrt_model_exec perf \
    --model_file ./ConvNeXt-deploy_224x224_nv12.bin \
    --thread_num 1
```

## Runtime Protocol

The generated deployment model is expected to follow this runtime protocol:

- Input tensor type: `NV12`
- Input resolution: `224x224`
- Output tensor shape: `1x1000x1x1`
- Output tensor type: `F32`

This protocol matches the Python runtime under [`../runtime/python/README.md`](../runtime/python/README.md).
