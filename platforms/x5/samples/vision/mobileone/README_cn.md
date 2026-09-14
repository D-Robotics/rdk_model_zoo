[English](./README.md) | 简体中文

# MobileOne 模型说明

本目录给出 MobileOne sample 在 Model Zoo 中的完整使用说明，包括算法概览、模型转换、运行时推理、模型文件管理和评测说明。

## 算法概述

MobileOne 是面向低时延边缘部署的轻量级 CNN 骨干网络。模型通过结构重参数化，在训练阶段保留多分支表达能力，在推理阶段收敛为更简洁的部署结构。

- **论文**: [MobileOne: An Improved One millisecond Mobile Backbone](http://arxiv.org/abs/2206.04040)
- **参考实现**: [apple/ml-mobileone](https://github.com/apple/ml-mobileone)

### 算法功能

MobileOne 支持以下任务：

- ImageNet 1000 类图像分类

### 算法特点

- **结构重参数化**：将训练期多分支结构融合为推理期友好的单路径结构。
- **低时延骨干**：面向移动端和嵌入式部署优化吞吐和延迟。
- **多尺度变体**：提供 `S0` 到 `S4` 五个已发布变体。
- **分类输出**：输出 ImageNet-1k 的 Top-K 类别 ID 和置信度分数。

![MobileOne 结构](./test_data/MobileOne_architecture.png)

## 目录结构

```text
.
|-- conversion
|   |-- MobileOne_S0_config.yaml
|   |-- MobileOne_S1_config.yaml
|   |-- MobileOne_S2_config.yaml
|   |-- MobileOne_S3_config.yaml
|   |-- MobileOne_S4_config.yaml
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
|       |-- mobileone.py
|       |-- README.md
|       |-- README_cn.md
|       `-- run.sh
|-- test_data
|   |-- ImageNet_1k.json
|   |-- inference.png
|   |-- MobileOne_architecture.png
|   `-- tiger_beetle.JPEG
|-- README.md
`-- README_cn.md
```

## 快速体验

### Python

- Python 详细说明请参考 [runtime/python/README_cn.md](./runtime/python/README_cn.md)。
- 快速体验命令：

```bash
cd runtime/python
bash run.sh
```

## 模型转换

- 预编译 `.bin` 模型通过 [model](./model/README_cn.md) 目录提供。
- 转换说明请参考 [conversion/README_cn.md](./conversion/README_cn.md)。

## 运行时推理

本 sample 当前维护的推理路径为 Python。

- Python 推理说明：[runtime/python/README_cn.md](./runtime/python/README_cn.md)

## 评测说明

评测说明、性能数据和验证结果请参考 [evaluator/README_cn.md](./evaluator/README_cn.md)。

## 性能数据

下表给出 `RDK X5` 上发布的 MobileOne 性能数据。

| 模型 | 输入尺寸 | 类别数 | 参数量 (M) | 浮点 Top-1 | 量化 Top-1 | 时延 (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MobileOne_S4 | 224x224 | 1000 | 14.8 | 78.75% | 76.50% | 4.58 | 256.52 |
| MobileOne_S3 | 224x224 | 1000 | 10.1 | 77.27% | 75.75% | 2.93 | 437.85 |
| MobileOne_S2 | 224x224 | 1000 | 7.8 | 74.75% | 71.25% | 2.11 | 653.68 |
| MobileOne_S1 | 224x224 | 1000 | 4.8 | 72.31% | 70.45% | 1.31 | 1066.95 |
| MobileOne_S0 | 224x224 | 1000 | 2.1 | 69.25% | 67.58% | 0.80 | 2453.17 |

![推理结果](./test_data/inference.png)

## License

遵循 Model Zoo 顶层 License。
