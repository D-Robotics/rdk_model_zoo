# MobileNetV2 模型说明

本目录描述 MobileNetV2 在本 Model Zoo 中的完整使用流程，包括：算法介绍、模型转换、运行时推理（Python/C++）、可复用的前后处理接口说明，以及模型评估步骤。

---

## 算法介绍（Algorithm Overview）

MobileNetV2 是一种专为移动端与嵌入式设备设计的轻量级卷积神经网络分类模型，具有以下特性：

- **倒置残差结构**：引入 Inverted Residuals 与 Linear Bottleneck，在保持精度的同时大幅降低计算量
- **深度可分离卷积**：将标准卷积分解为深度卷积与逐点卷积，显著减少参数量与计算量
- **轻量高效**：相比 ResNet 系列，参数量更少、推理速度更快，适合资源受限的嵌入式平台部署
- **通用性强**：在 ImageNet 上预训练，可直接用于 1000 类图像分类

### 算法功能
MobileNetV2 能完成以下任务：

- 单张图像的多类别分类
- 输出各类别的置信度分数及 Top-K 预测结果

### 原始资料
- MobileNetV2 论文：[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- PyTorch 官方实现：https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py

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
|   `-- download_model.sh               # HBM 模型下载脚本
|-- runtime                             # 模型推理示例
|   |-- cpp                             # C++ 推理工程
|   |   |-- inc                         # C++ 头文件
|   |   |   `-- mobilenetv2.hpp         # MobileNetV2 模型封装接口
|   |   |-- src                         # C++ 源码
|   |   |   |-- main.cpp                # 推理入口程序
|   |   |   `-- mobilenetv2.cpp         # MobileNetV2 推理实现
|   |   |-- CMakeLists.txt              # CMake 构建配置
|   |   |-- README.md                   # C++ 推理示例使用说明
|   |   `-- run.sh                      # C++ 示例运行脚本
|   `-- python                          # Python 推理示例
|       |-- README.md                   # Python 推理示例使用说明
|       |-- main.py                     # Python 推理入口脚本
|       |-- run.sh                      # Python 示例运行脚本
|       `-- mobilenetv2.py              # MobileNetV2 推理与后处理实现
|-- test_data                           # 测试数据
|   |-- zebra_cls.jpg                   # 示例测试图片
|   `-- imagenet1000_labels.txt         # ImageNet 1000 类别标签文件
`-- README.md                           # MobileNetV2 示例整体说明与快速指引
```

---

## 快速体验（QuickStart）

为了便于用户快速上手体验，每个模型均提供了`run.sh`脚本，用户运行此脚本即可一键运行相应模型，此脚本主要进行如下操作：
- 检测系统环境是否满足要求，若不满足则自动安装相应包；
- 检测推理所需的hbm模型文件是否存在，不存在则自动下载；
- 创建build目录，编译c++项目（仅C++项目）；
- 运行编译好的可执行文件或相应的python脚本；

### C++

- 进入`runtime`目录下的`cpp`目录，运行`run.sh`脚本，即可快速体验
    ```bash
    cd runtime/cpp/
    ./run.sh
    ```
- 若想了解`c++`代码的详细使用方法，或step by step运行模型请参考`runtime/cpp/README.md`；

### python

 - 进入`runtime`目录下的`python`目录，运行`run.sh`脚本，即可快速体验
    ```bash
    cd runtime/python/
    ./run.sh
    ```
- 若想了解`python`代码的详细使用方法，或step by step运行模型请参考`runtime/python/README.md`；

---

## 模型转换（Model Conversion）

- ModelZoo 已提供适配完成的 HBM 模型文件，用户可直接运行`model` 目录下的`download_model.sh`脚本下载并使用，如不关心模型转换流程，**可跳过本小节**。

- 如需自定义模型转换参数，或了解完整的模型转换流程，请参考`conversion/README.md`。

---

## 模型推理（Runtime）

MobileNetV2 模型推理示例同时提供 C++ 和 Python 两种实现方式，分别面向不同的使用场景与开发需求。两种版本在模型能力与推理结果上保持一致，但在使用方式和工程形态上有所区别。

### C++ 版本

    - 提供完整的工程化示例，适合集成到实际应用中;

    - 包含模型封装类、参数解析、推理流程及构建方式说明;

    - 具体编译、运行方式及接口说明请参考 `runtime/cpp/README.md`

### Python 版本

    - 以脚本形式提供，适合快速验证模型效果与算法流程;

    - 示例中展示了模型加载、推理执行、后处理以及结果可视化的完整过程;

    - 具体使用方法、参数说明及接口说明请参考 `runtime/python/README.md`;

---

## 模型评估（Evaluator）

`evaluator/` 用于模型精度、性能及数值一致性评估，详细说明请参考该目录。

---

## 推理结果

以 `zebra_cls.jpg` 为输入，Top-5 分类结果如下：

```bash
Top-5 Classification Results:
  [0] zebra: 0.9922
  [1] tiger, Panthera tigris: 0.0040
  [2] hartebeest: 0.0013
  [3] tiger cat: 0.0007
  [4] impala, Aepyceros melampus: 0.0005
```

---

## License
遵循 Model Zoo 顶层 License。
