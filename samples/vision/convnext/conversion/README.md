# Model Conversion

This directory describes how to convert the ConvNeXt model to the BPU-runnable BIN/HBM format.

## Directory Structure

```text
.
├── ConvNeXt_atto.yaml              # Atto model PTQ configuration
├── ConvNeXt_femto.yaml             # Femto model PTQ configuration
├── ConvNeXt_nano.yaml              # Nano model PTQ configuration
├── ConvNeXt_pico.yaml              # Pico model PTQ configuration
├── README.md                        # Documentation (English)
└── README_cn.md                     # Documentation (Chinese)
```

## Conversion Process

1. **Prepare Environment**: Install the RDK X5 OpenExplore toolchain.
2. **Export ONNX**: Export the pretrained model to ONNX format.
3. **PTQ Quantization**: Use `hb_mapper` with the provided YAML configurations.

For more details on the general conversion workflow, refer to the [Model Zoo Conversion Guide](../../../../../docs/README.md).
