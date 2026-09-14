[English](./README.md) | 简体中文

# 模型(EfficientSAM)

本目录下载并存储 EfficientSAM 编码器与解码器的预量化 HBM 模型。

## 目录结构

```text
.
├── download_model.sh      # 下载预编译模型
└── README.md              # 文档
```

## 下载模型

下载适配 RDK S100/S100P/S600 的预编译 EfficientSAM 模型：

```bash
bash download_model.sh            # 自动探测板卡
bash download_model.sh nash-e     # 显式 march
```

脚本自动探测板卡，将模型下载到 `./nash-e/`、`./nash-m/` 或 `./nash-p/`。

## 说明

- RDK S100/S100P/S600 的推理模型格式为 `.hbm`。
- EfficientSAM 采用双模型流水线：一个图像编码器加一个提示解码器，均按 march 分别编译。
- 模型后缀随平台不同：S100 用 `nashe`，S100P 用 `nashm`，S600 用 `nashp`。

## 已发布模型

| 板卡 | march | 编码器 `.hbm` | 解码器 `.hbm` |
|---|---|---|---|
| S100 | nash-e | `nash-e/efficient_sam_vitt_encoder_512x512_nashe.hbm` | `nash-e/efficient_sam_vitt_decoder_512_nashe.hbm` |
| S100P | nash-m | `nash-m/efficient_sam_vitt_encoder_512x512_nashm.hbm` | `nash-m/efficient_sam_vitt_decoder_512_nashm.hbm` |
| S600 | nash-p | `nash-p/efficient_sam_vitt_encoder_512x512_nashp.hbm` | `nash-p/efficient_sam_vitt_decoder_512_nashp.hbm` |

## License

本目录遵循 [Apache 2.0 License](../../../../LICENSE)。