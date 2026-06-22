[English](./README.md) | 简体中文

# SigLIP 模型下载说明

本目录用于存放 SigLIP HBM 模型。运行 `download_model.sh` 可下载默认模型或指定模型。

## 下载命令

```bash
cd samples/vision/siglip/model
bash download_model.sh
bash download_model.sh bpu-siglip-so400m-patch14-384
```

## 模型列表

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

下载基础路径：

```text
https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/
```

## License

本目录遵循 [Apache 2.0 License](../../../../LICENSE)。
