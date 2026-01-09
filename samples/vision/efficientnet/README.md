# EfficientNet 模型说明

## 简介 (Introduction)
EfficientNet 是一种卷积神经网络架构和缩放方法，它使用复合系数均匀地缩放深度、宽度和分辨率所有维度。相比于传统网络仅调整单一维度（如宽度、深度或分辨率），EfficientNet 通过复合缩放方法在给定的资源限制下最大化了网络性能。

本示例演示了如何使用 BPU 对 EfficientNet-Lite0 模型进行图像分类推理。

---

## 算法介绍 (Algorithm Overview)
EfficientNet 的核心思想是提出了一种复合缩放方法（Compound Scaling Method）。传统的网络设计通常固定其中两个维度，只调整第三个维度（例如 ResNet 增加深度），而 EfficientNet 论证了这三个维度之间是相互依赖的。例如，更高的分辨率需要更深的网络来增加感受野，同时也需要更宽的网络来捕获更细粒度的模式。

EfficientNet-Lite 系列是针对移动设备和物联网场景优化的版本，移除了对硬件不友好的 Squeeze-and-Excitation (SE) 模块，并将 Swish 激活函数替换为 ReLU6，以便于量化和在边缘设备上部署。

### 算法功能
EfficientNet-Lite0 模型能够在移动设备上高效地进行图像分类，具有以下功能：
- **高效性**：通过复合缩放方法，EfficientNet-Lite0 在保持较小模型体积的同时，提供了较高的准确率。
- **灵活性**：支持多种输入尺寸，用户可以根据实际需求选择合适的输入尺寸进行推理。
- **易用性**：提供简单易用的推理接口，用户只需提供输入图像即可获得分类结果。

### 原始资料

- 论文: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- 官方仓库: [tensorflow/tpu/models/official/efficientnet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)

---

## 目录结构 (Directory Structure)

本目录包含：

```bash
.
|-- 3rdparty                            # 第三方依赖或算法相关资源
|-- conversion                          # 模型转换流程
|   `-- README.md                       # 模型转换使用说明
|-- evaluator                           # 模型评估相关内容
|   `-- README.md                       # 模型评估说明
|-- model                               # 模型文件及下载脚本
|   `-- download_model.sh               # HBM 模型下载脚本
|   `-- README.md                       # HBM 模型下载使用说明
|-- runtime                             # 模型推理示例
|   |-- cpp                             # C++ 推理工程
|   `-- python                          # Python 推理示例
|       |-- README.md                   # Python 推理示例使用说明
|       |-- main.py                     # Python 推理入口脚本
|       |-- run.sh                      # Python 示例运行脚本
|       `-- efficient.py                # EfficientNet 推理与后处理实现
|-- test_data                           # 推理结果与示例数据
|   `-- result.jpg                      # 示例推理结果图像
`-- README.md                           # EfficientNet 示例整体说明与快速指引
```

---

## 快速体验 (QuickStart)

为了便于用户快速上手体验，每个模型均提供了`run.sh`脚本，用户运行此脚本即可一键运行相应模型，此脚本主要进行如下操作：
- 检测系统环境是否满足要求，若不满足则自动安装相应包；
- 检测推理所需的hbm模型文件是否存在，不存在则自动下载；
- 创建build目录，编译c++项目（仅C++项目）；
- 运行编译好的可执行文件或相应的python脚本；

### C++

### Python
进入 `runtime/python` 目录并运行脚本：
```bash
cd runtime/python
bash run.sh
```
- 该脚本会自动下载模型、安装必要的依赖（如果需要），并在示例图片上运行推理。若想了解`python`代码的详细使用方法，或step by step运行模型请参考`runtime/python/README.md`； [runtime/python/README.md](runtime/python/README.md)。

---

## 模型转换 (Model Conversion)

- ModelZoo 已提供适配完成的 HBM 模型文件，用户可直接运行`model` 目录下的`download_model.sh`脚本下载并使用，如不关心模型转换流程，**可跳过本小节**。

- 如需自定义模型转换参数，或了解完整的模型转换流程，请参考`conversion/README.md`。

---

## 模型推理 (Runtime)
本仓库提供 EfficientNet 的 Python 推理示例。详细的接口说明和运行方式请参考：
- [Python Runtime](runtime/python/README.md)

---

## 推理结果 (Inference Result)
在 RDK S100 平台上运行 EfficientNet-Lite0 模型，对 `Scottish_deerhound.JPEG` 进行分类的典型输出如下：

```bash
Top-5 Results:
  Scottish deerhound, deerhound: 0.8234
  greyhound: 0.0123
  Saluki, gazelle hound: 0.0056
  whippet: 0.0034
  Irish wolfhound: 0.0021
```

---

## License
遵循 ModelZoo 顶层 License。