English | [简体中文](./README_cn.md)

# DINOv2 Model Download Guide

## Download Commands

The model march is auto-detected from the on-board SoC. Run on the target
board:

```bash
cd samples/vision/dinov2/model
bash download_model.sh
```

To download a specific march on any machine:

```bash
bash download_model.sh nash-e   # RDK S100
bash download_model.sh nash-m   # RDK S100P
bash download_model.sh nash-p   # RDK S600
```

## Model List

| Model Name | Support BPU |
|---|---|
| `dinov2_vits14_224_int16_nashe.hbm` | Nash-E |
| `dinov2_vits14_224_int16_nashm.hbm` | Nash-M |
| `dinov2_vits14_224_int16_nashp.hbm` | Nash-P |

## License

This model is quantized from the Apache-2.0 licensed
[DINOv2](https://github.com/facebookresearch/dinov2) weights published by
Meta AI. See [../../../../LICENSE](../../../../LICENSE).
