[English](./README.md) | 简体中文

# MobileNetV3 模型说明

本目录给出 MobileNetV3 sample 在 Model Zoo 中的完整使用说明，包括算法概览、模型转换、运行时推理、模型文件管理和评测说明。

## 算法介绍

MobileNetV3 是一种通过 Network Architecture Search 和 NetAdapt 优化的轻量级卷积神经网络，用于高效图像分类。

- **论文**: [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
- **参考实现**: [timm/models/mobilenetv3.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mobilenetv3.py)

### 算法功能

MobileNetV3 支持以下任务：

- ImageNet 1000 类图像分类

### 算法特点

- **深度可分离卷积**：保留 MobileNet 系列的高效卷积结构。
- **倒残差结构**：使用 expansion-depthwise-projection 结构进行高效特征提取。
- **SE 注意力模块**：重新标定通道权重，提高特征表达能力。
- **H-Swish 激活函数**：使用硬件友好的激活函数，适合嵌入式部署。

![MobileNetV3 架构](./test_data/MobileNetV3_architecture.png)

## 目录结构

```text
.
|-- conversion
|   |-- MobileNetV3_config.yaml
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
|       |-- mobilenetv3.py
|       |-- README.md
|       |-- README_cn.md
|       `-- run.sh
|-- test_data
|   |-- ImageNet_1k.json
|   |-- inference.png
|   |-- kit_fox.JPEG
|   `-- MobileNetV3_architecture.png
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

## 模型推理

本 sample 当前维护的推理路径为 Python。

- Python 推理说明: [runtime/python/README_cn.md](./runtime/python/README_cn.md)

## 模型评估

评测说明、性能数据和验证结果请参考 [evaluator/README_cn.md](./evaluator/README_cn.md)。

## 性能数据

下表为 `RDK X5` 上发布的 MobileNetV3-Large 性能数据。

| 模型 | 尺寸 | 类别数 | 参数量 (M) | 浮点 Top-1 | 量化 Top-1 | 延迟 (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MobileNetV3-Large | 224x224 | 1000 | 5.5 | 74.8% | 64.8% | 2.02 | 714+ |

![推理结果](./test_data/inference.png)

## License

遵循 Model Zoo 顶层 License。
