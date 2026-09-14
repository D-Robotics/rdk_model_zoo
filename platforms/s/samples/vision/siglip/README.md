English | [简体中文](./README_cn.md)

# SigLIP Model Description

SigLIP is an image-text multimodal model family commonly used as the vision encoder in VLM and VLA models. This sample provides model download, runtime, and evaluation instructions for SigLIP vision encoder HBM models on RDK S100/S100P. The packed models expose both global image embeddings through `pooler_output` and patch-level visual features through `last_hidden_state`.

## Algorithm Overview

SigLIP uses independent image and text encoders to generate multimodal representations. Its vision encoder is usually ViT-based and produces high-dimensional image embeddings for downstream models such as PaliGemma, MiniCPM-V, RDT, PI0, and OpenVLA.

## Algorithm Capabilities

- Global image embedding through the `pooler_output` submodel.
- Visual token features through the `last_hidden_state` submodel.
- BPU-accelerated inference on RDK S100/S100P.
- Output shape, numeric range, mean, standard deviation, and L2 norm summary for runtime validation.

## Algorithm Features

- The model input is float32 NCHW RGB in the `[-1, 1]` range.
- Each HBM file contains two fixed submodels: `pooler_output` and `last_hidden_state`.
- Preprocessing uses aspect-ratio resize, gray padding, and `/127.5 - 1.0` normalization.

## Directory Structure

```text
siglip/
├── conversion/
├── evaluator/
├── model/
├── runtime/
│   └── python/
├── test_data/
├── README.md
└── README_cn.md
```

## Quick Start

```bash
cd samples/vision/siglip/runtime/python
bash run.sh
```

The default command downloads `bpu-siglip-base-patch16-224.hbm`, reads `test_data/dog.jpg`, runs the `pooler_output` submodel, and prints an output summary.

## Model Conversion

See [conversion/README.md](./conversion/README.md). This sample provides precompiled HBM models for RDK S100/S100P.

## Runtime

See [runtime/python/README.md](./runtime/python/README.md).

## Model Evaluation

See [evaluator/README.md](./evaluator/README.md) for performance, accuracy, and evaluation notes.

## Model List

The following prebuilt HBM models are available for download. Each packed model file contains both `pooler_output` and `last_hidden_state` submodels sharing the same weights.

| Model Name | Download | Supported BPU |
|---|---|---|
| bpu-siglip-base-patch16-224 | [bpu-siglip-base-patch16-224.hbm](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-base-patch16-224.hbm) | Nash-e, Nash-m |
| bpu-siglip-base-patch16-384 | [bpu-siglip-base-patch16-384.hbm](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-base-patch16-384.hbm) | Nash-e, Nash-m |
| bpu-siglip-base-patch16-512 | [bpu-siglip-base-patch16-512.hbm](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-base-patch16-512.hbm) | Nash-e, Nash-m |
| bpu-siglip-large-patch16-256 | [bpu-siglip-large-patch16-256.hbm](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-large-patch16-256.hbm) | Nash-e, Nash-m |
| bpu-siglip-large-patch16-384 | [bpu-siglip-large-patch16-384.hbm](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-large-patch16-384.hbm) | Nash-e, Nash-m |
| bpu-siglip-so400m-patch14-224 | [bpu-siglip-so400m-patch14-224.hbm](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-so400m-patch14-224.hbm) | Nash-e, Nash-m |
| bpu-siglip-so400m-patch14-384 | [bpu-siglip-so400m-patch14-384.hbm](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-so400m-patch14-384.hbm) | Nash-e, Nash-m |
| bpu-siglip-so400m-patch16-256-i18n | [bpu-siglip-so400m-patch16-256-i18n.hbm](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-so400m-patch16-256-i18n.hbm) | Nash-e, Nash-m |

See [model/README.md](./model/README.md) for download script usage.

## Contributors

Cauchy @吴超

## License

This sample is licensed under the [Apache 2.0 License](../../../LICENSE).
