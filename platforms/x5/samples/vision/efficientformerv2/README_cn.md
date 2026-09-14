[English](./README.md) | 简体中文

# EfficientFormerV2 模型说明

本目录给出 EfficientFormerV2 sample 在 Model Zoo 中的完整使用说明，包括算法概览、模型转换、运行时推理、模型文件管理和评测说明。

## 算法概述

EfficientFormerV2 是面向移动端的视觉 Transformer 模型系列。该模型使用混合视觉骨干网络，并通过细粒度联合搜索策略在模型尺寸、延迟和精度之间取得平衡，适合边缘侧部署。

- **论文**：[EfficientFormerV2: Rethinking Vision Transformers for MobileNet Size and Speed](https://arxiv.org/abs/2212.08059)
- **参考实现**：[snap-research/EfficientFormer](https://github.com/snap-research/EfficientFormer)

### 算法功能

EfficientFormerV2 支持以下任务：

- ImageNet 1000 类图像分类

### 算法特点

- **面向移动端的骨干网络**：使用混合骨干结构提升边缘侧分类推理效率。
- **联合搜索策略**：在架构搜索中同时优化延迟和参数量。
- **分层结构设计**：采用四阶段结构，特征尺寸分别为输入分辨率的 `1/4`、`1/8`、`1/16` 和 `1/32`。
- **边缘侧部署**：提供 S0、S1、S2 三个 RDK X5 部署模型，输入格式为 packed NV12。

![EfficientFormerV2 架构](./test_data/EfficientFormerV2_architecture.png)

## 目录结构

```text
.
|-- conversion
|   |-- EfficientFormerv2_s0_config.yaml
|   |-- EfficientFormerv2_s1_config.yaml
|   |-- EfficientFormerv2_s2_config.yaml
|   |-- README.md
|   `-- README_cn.md
|-- evaluator
|   |-- README.md
|   `-- README_cn.md
|-- model
|   |-- download.sh
|   |-- README.md
|   `-- README_cn.md
|-- runtime
|   `-- python
|       |-- main.py
|       |-- efficientformerv2.py
|       |-- README.md
|       |-- README_cn.md
|       `-- run.sh
|-- test_data
|   |-- EfficientFormerV2_architecture.png
|   |-- ImageNet_1k.json
|   |-- goldfish.JPEG
|   `-- inference.png
|-- README.md
`-- README_cn.md
```

## 快速开始

### Python

- 详细 Python 使用方式请参考 [runtime/python/README_cn.md](./runtime/python/README_cn.md)。
- 快速体验：

```bash
cd runtime/python
bash run.sh
```

## 模型转换

- 预编译 `.bin` 模型文件由 [model](./model/README_cn.md) 目录提供。
- 转换说明见 [conversion/README_cn.md](./conversion/README_cn.md)。

## 运行时推理

本 sample 维护的推理路径为 Python。

- Python 运行说明：[runtime/python/README_cn.md](./runtime/python/README_cn.md)

## 模型评测

评测说明、性能数据和验证摘要见 [evaluator/README_cn.md](./evaluator/README_cn.md)。

## 性能数据

下表为 EfficientFormerV2 在 `RDK X5` 上的公开性能数据。

| 模型 | 尺寸 | 类别数 | 参数量 (M) | Float Top-1 | Quant Top-1 | 延迟 (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientFormerV2-S2 | 224x224 | 1000 | 12.6 | 77.50% | 70.75% | 6.99 | 152.40 |
| EfficientFormerV2-S1 | 224x224 | 1000 | 6.1 | 77.25% | 68.75% | 4.24 | 275.95 |
| EfficientFormerV2-S0 | 224x224 | 1000 | 3.5 | 74.25% | 68.50% | 5.79 | 198.45 |

![推理结果](./test_data/inference.png)

## 许可证

遵循 Model Zoo 顶层 License。
