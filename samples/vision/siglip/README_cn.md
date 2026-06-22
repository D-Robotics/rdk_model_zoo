[English](./README.md) | 简体中文

# SigLIP 模型说明

SigLIP 是一个图像-文本多模态模型家族，常作为 VLM 和 VLA 模型的视觉编码器使用。本 sample 提供 RDK S100/S100P 上 SigLIP 视觉编码器 HBM 模型的下载、运行和评测说明，输出包括图像全局嵌入 `pooler_output` 和 patch 级视觉特征 `last_hidden_state`。

## 算法介绍

SigLIP 使用独立的图像编码器和文本编码器生成两种模态的表示。视觉编码部分通常基于 ViT 结构，能够将图像编码为高维嵌入向量，供 PaliGemma、MiniCPM-V、RDT、PI0、OpenVLA 等下游模型使用。

## 算法功能

- 图像全局嵌入：运行 `pooler_output` 子模型。
- 视觉 token 特征：运行 `last_hidden_state` 子模型。
- S100/S100P BPU 加速推理。
- 输出形状、数值范围、均值、标准差和 L2 范数摘要，便于快速确认结果有效性。

## 算法特点

- 模型输入为 float32 NCHW RGB，取值范围 `[-1, 1]`。
- HBM 模型内包含 `pooler_output` 和 `last_hidden_state` 两个固定子模型。
- 预处理采用等比例缩放、灰色 padding、`/127.5 - 1.0` 归一化。

## 目录结构

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

## 快速体验

```bash
cd samples/vision/siglip/runtime/python
bash run.sh
```

默认命令会下载 `bpu-siglip-base-patch16-224.hbm`，读取 `test_data/dog.jpg`，运行 `pooler_output` 子模型并打印输出摘要。

## 模型转换

SigLIP 的转换说明见 [conversion/README_cn.md](./conversion/README_cn.md)。当前 sample 提供预编译 HBM 模型，适配 RDK S100/S100P。

## 模型推理

Python 运行说明见 [runtime/python/README_cn.md](./runtime/python/README_cn.md)。

## 模型评估

性能、精度和评测方式见 [evaluator/README_cn.md](./evaluator/README_cn.md)。

## 模型列表

以下预编译 HBM 模型可供下载。每个模型文件内包含 `pooler_output` 和 `last_hidden_state` 两个共享权重的子模型。

| 模型名称 | 下载 | 支持的 BPU |
|---|---|---|
| bpu-siglip-base-patch16-224 | [bpu-siglip-base-patch16-224.hbm](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-base-patch16-224.hbm) | Nash-e, Nash-m |
| bpu-siglip-base-patch16-384 | [bpu-siglip-base-patch16-384.hbm](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-base-patch16-384.hbm) | Nash-e, Nash-m |
| bpu-siglip-base-patch16-512 | [bpu-siglip-base-patch16-512.hbm](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-base-patch16-512.hbm) | Nash-e, Nash-m |
| bpu-siglip-large-patch16-256 | [bpu-siglip-large-patch16-256.hbm](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-large-patch16-256.hbm) | Nash-e, Nash-m |
| bpu-siglip-large-patch16-384 | [bpu-siglip-large-patch16-384.hbm](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-large-patch16-384.hbm) | Nash-e, Nash-m |
| bpu-siglip-so400m-patch14-224 | [bpu-siglip-so400m-patch14-224.hbm](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-so400m-patch14-224.hbm) | Nash-e, Nash-m |
| bpu-siglip-so400m-patch14-384 | [bpu-siglip-so400m-patch14-384.hbm](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-so400m-patch14-384.hbm) | Nash-e, Nash-m |
| bpu-siglip-so400m-patch16-256-i18n | [bpu-siglip-so400m-patch16-256-i18n.hbm](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP/bpu-siglip-so400m-patch16-256-i18n.hbm) | Nash-e, Nash-m |

下载脚本用法见 [model/README_cn.md](./model/README_cn.md)。

## 贡献者

Cauchy @吴超

## License

本 sample 遵循 [Apache 2.0 License](../../../LICENSE)。
