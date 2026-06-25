English | [简体中文](./README_cn.md)

# EfficientNet-Lite Models

Use `download_model.sh` to download published HBM models into this sample
directory.  The script auto-detects the target SoC from the board (or accepts
it as the first argument) and fetches the correct prebuilt variant.

## Usage

```bash
# Auto-detect SoC, download all variants
bash download_model.sh

# Or specify target SoC explicitly
bash download_model.sh s100 lite0
bash download_model.sh s600 all
```

## Published Models

| SoC   | URL |
|-------|-----|
| `s100` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/EfficientNet/efficientnet_lite*_*x*_nv12.hbm` |
| `s600` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/EfficientNet/efficientnet_lite*_*x*_nv12.hbm` |

## Variants

| Variant | S100 file | S600 file |
| --- | --- | --- |
| `lite0` | `s100/efficientnet_lite0_224x224_nv12.hbm` | `s600/efficientnet_lite0_224x224_nv12.hbm` |
| `lite1` | `s100/efficientnet_lite1_240x240_nv12.hbm` | `s600/efficientnet_lite1_240x240_nv12.hbm` |
| `lite2` | `s100/efficientnet_lite2_260x260_nv12.hbm` | `s600/efficientnet_lite2_260x260_nv12.hbm` |
| `lite3` | `s100/efficientnet_lite3_300x300_nv12.hbm` | `s600/efficientnet_lite3_300x300_nv12.hbm` |
| `lite4` | `s100/efficientnet_lite4_380x380_nv12.hbm` | `s600/efficientnet_lite4_380x380_nv12.hbm` |