English | [简体中文](./README_cn.md)

# SigLIP Model Download Guide

This directory stores SigLIP HBM models. Run `download_model.sh` to download the default model or a specified model.

## Download Commands

```bash
cd samples/vision/siglip/model
bash download_model.sh
bash download_model.sh bpu-siglip-so400m-patch14-384
```

## Model List

| Model Name | Support BPU |
|---|---|
| `bpu-siglip-base-patch16-224.hbm` | Nash-E / Nash-M |
| `bpu-siglip-base-patch16-384.hbm` | Nash-E / Nash-M |
| `bpu-siglip-base-patch16-512.hbm` | Nash-E / Nash-M |
| `bpu-siglip-large-patch16-256.hbm` | Nash-E / Nash-M |
| `bpu-siglip-large-patch16-384.hbm` | Nash-E / Nash-M |
| `bpu-siglip-so400m-patch14-224.hbm` | Nash-E / Nash-M |
| `bpu-siglip-so400m-patch14-384.hbm` | Nash-E / Nash-M |
| `bpu-siglip-so400m-patch16-256-i18n.hbm` | Nash-E / Nash-M |

Base download path:

```text
https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/
```

## License

This directory is licensed under the [Apache 2.0 License](../../../../LICENSE).
