[English](./README.md) | 简体中文

# EfficientNet-Lite 模型

使用 `download_model.sh` 下载已发布的 HBM 模型到
`/opt/hobot/model/<soc>/basic/`。脚本会自动检测目标 SoC（也可通过第一参数
指定）并下载对应的预编译模型。下载路径与 runtime 示例脚本以及
`EfficientNetConfig` 默认 `model_path` 保持一致，所有入口都从同一位置加载
模型。

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

执行 `bash download_model.sh <soc> all` 后，文件位于：

```text
/opt/hobot/model/<soc>/basic/efficientnet_lite0_224x224_nv12.hbm
/opt/hobot/model/<soc>/basic/efficientnet_lite1_240x240_nv12.hbm
/opt/hobot/model/<soc>/basic/efficientnet_lite2_260x260_nv12.hbm
/opt/hobot/model/<soc>/basic/efficientnet_lite3_300x300_nv12.hbm
/opt/hobot/model/<soc>/basic/efficientnet_lite4_380x380_nv12.hbm
```

`<soc>` 为 `s100` 或 `s600`，已存在的文件会跳过。