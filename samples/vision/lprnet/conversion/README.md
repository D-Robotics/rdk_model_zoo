# LPRNet Model Conversion

This directory records the conversion-side notes for LPRNet on RDK X5.

For the complete model conversion workflow, refer directly to the OE package.

## Available Assets

The repository currently provides:

- the released `.bin` model used by this sample
- input and output tensor protocol notes

The repository does **not** provide a full in-repo conversion toolchain.

## Supported X5 Model

| Model | Input Size | Runtime Format |
| --- | --- | --- |
| `lpr.bin` | `1x3x24x94` | `.bin` |

## `hb_mapper checker`

After preparing the conversion assets according to the OE package, use:

```bash
hb_mapper checker --model-type onnx --config your_lprnet_config.yaml
```

## `hb_mapper makertbin`

Generate the deployable `.bin` model:

```bash
hb_mapper makertbin --model-type onnx --config your_lprnet_config.yaml
```

## `hrt_model_exec`

Inspect model inputs and outputs on board:

```bash
hrt_model_exec model_info --model_file lpr.bin
```

## Output Protocol

The X5 LPRNet model used by this sample follows the original demo protocol:

- input tensor: `1 x 3 x 24 x 94`, `float32`, `NCHW`
- output tensor: `1 x 68 x 18`, `float32`, `NCHW`

The Python runtime decodes the final character sequence with a CTC-style deduplication rule.
