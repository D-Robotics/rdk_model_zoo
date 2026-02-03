# EfficientNet 模型说明

本目录描述 EfficientNet 在本 Model Zoo 中的完整使用流程，包括：算法介绍、模型转换、运行时推理（Python）、可复用的前后处理接口说明，以及模型评估步骤。

---

## 算法介绍（Algorithm Overview）

EfficientNet 是一种卷积神经网络架构和缩放方法，它使用复合系数均匀地缩放深度、宽度和分辨率所有维度。EfficientNet-Lite 系列是针对移动设备和物联网场景优化的版本。本示例集成了 EfficientNet-Lite0 到 Lite4 共五个版本的量化模型。

### 算法功能
EfficientNet-Lite 系列模型能够在移动设备上高效地进行图像分类，具有以下功能：
- 图像分类
- 输出 Top-K 类别及置信度

### 原始资料
- 论文: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- 官方仓库: [tensorflow/tpu/models/official/efficientnet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)

---

## 目录结构（Directory Structure）

本目录包含：

```bash
.
|-- conversion                          # 模型转换流程
|   `-- README.md                       # 模型转换使用说明
|-- evaluator                           # 模型评估相关内容
|   `-- README.md                       # 模型评估说明
|-- model                               # 模型文件及下载脚本
|   |-- download_model.sh               # HBM 模型下载脚本 (支持 Lite0~Lite4)
|   `-- README.md                       # HBM 模型下载使用说明
|-- runtime                             # 模型推理示例
|   `-- python                          # Python 推理示例
|       |-- README.md                   # Python 推理示例使用说明
|       |-- main.py                     # Python 推理入口脚本
|       |-- run.sh                      # Python 示例运行脚本
|       `-- efficientnet.py             # EfficientNet 推理与后处理实现
`-- README.md                           # EfficientNet 示例整体说明与快速指引
```

---

## 快速体验（QuickStart）

为了便于用户快速上手体验，每个模型均提供了`run.sh`脚本，用户运行此脚本即可一键运行相应模型，此脚本主要进行如下操作：
- 检测系统环境是否满足要求，若不满足则自动安装相应包；
- 检测推理所需的hbm模型文件是否存在，不存在则自动下载；
- 运行相应的python脚本；

### C++

- 暂未提供 C++ 版本推理示例。

### python

 - 进入`runtime`目录下的`python`目录，运行`run.sh`脚本，即可快速体验
    ```bash
    cd runtime/python/
    ./run.sh
    ```
- 若想了解`python`代码的详细使用方法，或step by step运行模型请参考 `runtime/python/README.md`；

---

## 模型转换（Model Conversion）

- ModelZoo 已提供适配完成的 HBM 模型文件，用户可直接运行`model` 目录下的`download_model.sh`脚本下载并使用，如不关心模型转换流程，**可跳过本小节**。

- 如需自定义模型转换参数，或了解完整的模型转换流程，请参考`conversion/README.md`。

---

## 模型推理（Runtime）

EfficientNet 模型推理示例目前仅提供 Python 实现方式。

### C++ 版本

    - 暂未提供。

### Python 版本

    - 以脚本形式提供，适合快速验证模型效果与算法流程;

    - 示例中展示了模型加载、推理执行、后处理的完整过程;

    - 具体使用方法、参数说明及接口说明请参考 `runtime/python/README.md`;

---

## 模型评估（Evaluator）

`evaluator/` 用于模型精度、性能及数值一致性评估，详细说明请参考该目录。

---

## 推理结果

在 RDK S100 平台上运行 EfficientNet-Lite0 模型，对 `scottish_deerhound.JPEG` (位于 `datasets/imagenet/asset/`) 进行分类的典型输出如下：


---

## License
遵循 Model Zoo 顶层 License。
