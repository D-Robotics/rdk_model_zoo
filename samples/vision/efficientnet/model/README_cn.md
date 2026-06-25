[English](./README.md) | 简体中文

# EfficientNet-Lite 模型

使用 `download_model.sh` 下载已发布的 HBM 模型到当前 sample 目录。
脚本会自动检测目标 SoC（也可通过第一参数指定）并下载对应的预编译模型。

## 用法

```bash
# 自动检测 SoC，下载全部变体
bash download_model.sh

# 或明确指定 SoC 和变体
bash download_model.sh s100 lite0
bash download_model.sh s600 all
```

## 已发布模型

| SoC   | 下载地址 |
|-------|---------|
| `s100` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/EfficientNet/efficientnet_lite*_*x*_nv12.hbm` |
| `s600` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/EfficientNet/efficientnet_lite*_*x*_nv12.hbm` |

## 变体列表

| 变体 | S100 文件 | S600 文件 |
| --- | --- | --- |
| `lite0` | `s100/efficientnet_lite0_224x224_nv12.hbm` | `s600/efficientnet_lite0_224x224_nv12.hbm` |
| `lite1` | `s100/efficientnet_lite1_240x240_nv12.hbm` | `s600/efficientnet_lite1_240x240_nv12.hbm` |
| `lite2` | `s100/efficientnet_lite2_260x260_nv12.hbm` | `s600/efficientnet_lite2_260x260_nv12.hbm` |
| `lite3` | `s100/efficientnet_lite3_300x300_nv12.hbm` | `s600/efficientnet_lite3_300x300_nv12.hbm` |
| `lite4` | `s100/efficientnet_lite4_380x380_nv12.hbm` | `s600/efficientnet_lite4_380x380_nv12.hbm` |