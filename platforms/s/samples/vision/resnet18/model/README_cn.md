[English](./README.md) | 简体中文

# ResNet18 模型下载

在本目录运行下载脚本，并通过参数指定目标 SoC（`s100` 或 `s600`）：

```bash
# RDK S100
bash download_model.sh s100

# RDK S600
bash download_model.sh s600
```

脚本会将 HBM 文件下载到 `./<soc>/`：

| SoC | 本地路径 | URL |
| --- | --- | --- |
| `s100` | `./s100/resnet18_224x224_nv12.hbm` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/resnet18_224x224_nv12.hbm` |
| `s600` | `./s600/resnet18_224x224_nv12.hbm` | `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/ResNet/resnet18_224x224_nv12.hbm` |

本 sample 使用公开 RDK ResNet18 HBM 模型，S100/S600 模型文件名相同，仅 archive 子目录不同。
