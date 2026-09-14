[English](./README.md) | 简体中文

# ResNet 模型说明

本目录给出 ResNet sample 在 Model Zoo 中的完整使用说明，包括算法概览、模型转换、运行时推理、模型文件管理和评测说明。

## 算法介绍

ResNet 是一种引入残差学习和快捷连接的卷积神经网络，使深层网络训练变得可行且稳定。

- **论文**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **官方实现**: [pytorch/vision/models/resnet.py](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)

### 算法功能

ResNet 可完成以下任务：

- ImageNet 1000 类图像分类

### 算法特性

- **残差学习**：残差块让恒等映射更容易学习，稳定深层网络优化。
- **快捷连接**：跳跃连接有效缓解深层 CNN 的退化问题。
- **分类输出**：模型输出 Top-K 类别及对应置信度。

![ResNet 架构](./test_data/ResNet_architecture.png)

## 目录结构

```text
.
├── conversion
│   ├── README.md
│   └── README_cn.md
├── evaluator
│   ├── README.md
│   └── README_cn.md
├── model
│   ├── download.sh
│   ├── README.md
│   └── README_cn.md
├── runtime
│   └── python
│       ├── main.py
│       ├── README.md
│       ├── README_cn.md
│       ├── resnet.py
│       └── run.sh
├── test_data
│   ├── ImageNet_1k.json
│   ├── inference.png
│   ├── ResNet_architecture.png
│   ├── ResNet_architecture2.png
│   └── white_wolf.JPEG
├── README.md
└── README_cn.md
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
- 如需查看转换参考，请阅读 [conversion/README_cn.md](./conversion/README_cn.md)。

## 模型推理

本 sample 当前维护的推理路径为 Python。

- Python 推理说明: [runtime/python/README_cn.md](./runtime/python/README_cn.md)

## 模型评估

评测说明、性能数据和验证结果请参考 [evaluator/README_cn.md](./evaluator/README_cn.md)。

## 性能数据

下表为 `RDK X5` 上发布的 ResNet18 性能数据。

| 模型 | 尺寸 | 类别数 | 参数量 (M) | 浮点 Top-1 | 量化 Top-1 | 延迟 (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet18 | 224x224 | 1000 | 11.2 | 71.5% | 70.5% | 2.95 | 449+ |

![推理结果](./test_data/inference.png)

## License

遵循 Model Zoo 顶层 License。
