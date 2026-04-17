# FCOS Model Conversion

This directory records the FCOS conversion-related assets for RDK X5.

For the complete FCOS model conversion workflow, refer directly to the OE package.

## Available Assets

The repository currently provides:

- `hb_perf` result snapshots for the published X5 models
- model names and input resolutions for the released `.bin` files
- runtime protocol notes for the FCOS outputs

The repository does **not** provide a full in-repo conversion toolchain.
If you need to regenerate the model, follow the OE package workflow to prepare the required ONNX file and conversion configuration.

## Supported X5 Models

| Model | Input Size | Runtime Format |
| --- | --- | --- |
| `fcos_efficientnetb0_detect_512x512_bayese_nv12.bin` | 512x512 | `.bin` |
| `fcos_efficientnetb2_detect_768x768_bayese_nv12.bin` | 768x768 | `.bin` |
| `fcos_efficientnetb3_detect_896x896_bayese_nv12.bin` | 896x896 | `.bin` |


## `hb_mapper makertbin`

Generate the deployable `.bin` model:

```bash
hb_mapper makertbin --model-type onnx --config your_fcos_config.yaml
```

## `hb_perf`

Inspect the converted `.bin` model:

```bash
hb_perf fcos_efficientnetb0_detect_512x512_bayese_nv12.bin
hb_perf fcos_efficientnetb2_detect_768x768_bayese_nv12.bin
hb_perf fcos_efficientnetb3_detect_896x896_bayese_nv12.bin
```

Reference outputs:

![FCOS EfficientNet-B0 hb_perf](./fcos_efficientnetb0_512x512_nv12.png)
![FCOS EfficientNet-B2 hb_perf](./fcos_efficientnetb2_768x768_nv12.png)
![FCOS EfficientNet-B3 hb_perf](./fcos_efficientnetb3_896x896_nv12.png)

## `hrt_model_exec`

Inspect model inputs and outputs on board:

```bash
hrt_model_exec model_info --model_file fcos_efficientnetb0_detect_512x512_bayese_nv12.bin
```

## Output Protocol

The X5 FCOS model follows the original FCOS demo protocol:

- 5 classification outputs
- 5 box regression outputs
- 5 center-ness outputs

The Python runtime reorders these outputs by their fixed tensor shapes before decoding.
