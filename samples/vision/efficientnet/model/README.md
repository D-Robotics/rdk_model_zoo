English | [简体中文](./README_cn.md)

# EfficientNet-Lite Models

Use `download_model.sh` to download published HBM models to
`/opt/hobot/model/<soc>/basic/`. The script auto-detects the target SoC from
the board (or accepts it as the first argument) and fetches the correct
prebuilt variant. The download path matches the runtime sample scripts and
the default `model_path` resolved by `EfficientNetConfig`, so all entry
points read from the same location.

## Usage

```bash
# Auto-detect SoC, download all variants
bash download_model.sh

# Or specify target SoC and variant
bash download_model.sh s100 lite0
bash download_model.sh s600 all
```

## Published Models

| SoC   | URL |
|-------|-----|
| `s100` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/EfficientNet/efficientnet_lite*_*x*_nv12.hbm` |
| `s600` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/EfficientNet/efficientnet_lite*_*x*_nv12.hbm` |

## Variants

After running `bash download_model.sh <soc> all`, files land at:

```text
/opt/hobot/model/<soc>/basic/efficientnet_lite0_224x224_nv12.hbm
/opt/hobot/model/<soc>/basic/efficientnet_lite1_240x240_nv12.hbm
/opt/hobot/model/<soc>/basic/efficientnet_lite2_260x260_nv12.hbm
/opt/hobot/model/<soc>/basic/efficientnet_lite3_300x300_nv12.hbm
/opt/hobot/model/<soc>/basic/efficientnet_lite4_380x380_nv12.hbm
```

`<soc>` is `s100` or `s600`. Files that already exist are skipped.