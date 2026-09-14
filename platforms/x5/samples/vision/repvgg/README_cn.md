[English](./README.md) | 简体中文

# RepVGG 模型说明

本目录给出 RepVGG sample 在 Model Zoo 中的完整使用说明，包括算法概览、模型转换、运行时推理、模型文件管理和评测说明。

## 算法概述

RepVGG 是一种 VGG 风格的卷积神经网络家族，核心思想是结构重参数化。训练阶段可以使用多分支结构，部署阶段转换为由 `3x3` 卷积和 ReLU 组成的直连结构，以提升推理效率。

- **论文**: [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)
- **参考实现**: [DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)

### 算法功能

RepVGG 支持以下任务：

- ImageNet 1000 类图像分类

### 算法特点

- **直连推理结构**：部署后采用 VGG 风格卷积堆叠。
- **结构重参数化**：将训练期分支转换为部署期卷积层。
- **硬件友好**：主要由卷积和 ReLU 组成，适合边缘端推理。
- **多规格变体**：提供 `A0`、`A1`、`A2`、`B0`、`B1g2`、`B1g4` 等规格。

![RepVGG 结构](./test_data/RepVGG_architecture.png)

## 目录结构

```text
.
|-- conversion
|   |-- RepVGG_A0_config.yaml
|   |-- RepVGG_A1_config.yaml
|   |-- RepVGG_A2_config.yaml
|   |-- RepVGG_B0_config.yaml
|   |-- RepVGG_B1g2_config.yaml
|   |-- RepVGG_B1g4_config.yaml
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
|       |-- repvgg.py
|       |-- README.md
|       |-- README_cn.md
|       `-- run.sh
|-- test_data
|   |-- gooze.JPEG
|   |-- ImageNet_1k.json
|   |-- inference.png
|   `-- RepVGG_architecture.png
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

下表给出 `RDK X5` 上发布的 RepVGG 性能数据。

| 模型 | 输入尺寸 | 类别数 | 参数量 (M) | 浮点 Top-1 | 量化 Top-1 | 延迟 (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RepVGG_B1g2 | 224x224 | 1000 | 41.36 | 77.78 | 68.25 | 9.77 | 109.61 |
| RepVGG_B1g4 | 224x224 | 1000 | 36.12 | 77.58 | 62.75 | 7.58 | 144.39 |
| RepVGG_B0 | 224x224 | 1000 | 14.33 | 75.14 | 60.36 | 3.07 | 410.55 |
| RepVGG_A2 | 224x224 | 1000 | 25.49 | 76.48 | 62.97 | 6.07 | 186.04 |
| RepVGG_A1 | 224x224 | 1000 | 12.78 | 74.46 | 62.78 | 2.67 | 482.20 |
| RepVGG_A0 | 224x224 | 1000 | 8.30 | 72.41 | 51.75 | 1.85 | 757.73 |

![推理结果](./test_data/inference.png)

## License

遵循 Model Zoo 顶层 License。
