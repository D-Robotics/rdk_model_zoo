# UnetMobileNet 模型说明

本目录描述 UnetMobileNet 在本 Model Zoo 中的完整使用流程，包括：算法介绍、模型转换、运行时推理（Python/C++）、前后处理接口说明，以及模型评估步骤。

---

## 算法介绍（Algorithm Overview）

UnetMobileNet 是一种语义分割模型，采用 U-Net 编解码架构与 MobileNet 轻量化骨干网络，具有以下特性：

- **编解码结构（U-Net）**：通过跳跃连接融合浅层与深层特征，提升分割精度
- **轻量化骨干（MobileNet）**：采用深度可分离卷积，有效降低计算量
- **语义分割**：输出逐像素类别标签，支持 Cityscapes 19 类场景理解
- **适合端侧部署**：在 RDK S100/S600 等嵌入式平台上高效运行

### 算法功能

UnetMobileNet 能完成以下任务：

- 图像逐像素语义分割
- 输出类别 ID 热图（可视化为彩色叠加图）

### 原始资料

- U-Net 论文：[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- MobileNet 论文：[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- 数据集：[Cityscapes Dataset](https://www.cityscapes-dataset.com/)

---

## 目录结构（Directory Structure）

本目录包含：

```bash
.
|-- conversion                              # 模型转换流程
|   `-- README.md                           # 模型转换使用说明
|-- evaluator                               # 模型评估相关内容
|   `-- README.md                           # 模型评估说明
|-- model                                   # 模型文件及下载脚本
|   |-- download_model.sh                   # HBM 模型下载脚本
|   `-- README.md                           # 模型下载说明
|-- runtime                                 # 模型推理示例
|   |-- cpp                                 # C++ 推理工程
|   |   |-- inc                             # C++ 头文件
|   |   |   `-- unetmobilenet.hpp           # UnetMobileNet 模型封装接口
|   |   |-- src                             # C++ 源码
|   |   |   |-- main.cpp                    # 推理入口程序
|   |   |   `-- unetmobilenet.cpp           # UnetMobileNet 推理实现
|   |   |-- CMakeLists.txt                  # CMake 构建配置
|   |   |-- README.md                       # C++ 推理示例使用说明
|   |   `-- run.sh                          # C++ 示例运行脚本
|   `-- python                              # Python 推理示例
|       |-- README.md                       # Python 推理示例使用说明
|       |-- main.py                         # Python 推理入口脚本
|       |-- run.sh                          # Python 示例运行脚本
|       `-- unetmobilenet.py                # UnetMobileNet 推理与后处理实现
|-- test_data                               # 测试数据与推理结果
|   |-- segmentation.png                    # 示例测试图片（Cityscapes 场景）
|   `-- result.jpg                          # 示例推理结果图像
`-- README.md                               # UnetMobileNet 示例整体说明与快速指引
```

---

## 快速体验（QuickStart）

为了便于用户快速上手体验，每个模型均提供了 `run.sh` 脚本，用户运行此脚本即可一键运行相应模型，此脚本主要进行如下操作：
- 检测系统环境是否满足要求，若不满足则自动安装相应包；
- 检测推理所需的 hbm 模型文件是否存在，不存在则自动下载；
- 创建 build 目录，编译 C++ 项目（仅 C++ 项目）；
- 运行编译好的可执行文件或相应的 Python 脚本；

### C++

- 进入 `runtime` 目录下的 `cpp` 目录，运行 `run.sh` 脚本，即可快速体验
    ```bash
    cd runtime/cpp/
    ./run.sh
    ```
- 若想了解 C++ 代码的详细使用方法，或 step by step 运行模型请参考 `runtime/cpp/README.md`；

### Python

- 进入 `runtime` 目录下的 `python` 目录，运行 `run.sh` 脚本，即可快速体验
    ```bash
    cd runtime/python/
    ./run.sh
    ```
- 若想了解 Python 代码的详细使用方法，或 step by step 运行模型请参考 `runtime/python/README.md`；

---

## 模型转换（Model Conversion）

- ModelZoo 已提供适配完成的 HBM 模型文件，用户可直接运行 `model` 目录下的 `download_model.sh` 脚本下载并使用，如不关心模型转换流程，**可跳过本小节**。

- 自定义转换流程说明暂未提供，`conversion/` 目录内容待补充。

---

## 模型推理（Runtime）

UnetMobileNet 模型推理示例同时提供 C++ 和 Python 两种实现方式，分别面向不同的使用场景与开发需求。两种版本在模型能力与推理结果上保持一致，但在使用方式和工程形态上有所区别。

### C++ 版本

- 提供完整的工程化示例，适合集成到实际应用中；
- 包含模型封装类、参数解析、推理流程及构建方式说明；
- 具体编译、运行方式及接口说明请参考 `runtime/cpp/README.md`；

### Python 版本

- 以脚本形式提供，适合快速验证模型效果与算法流程；
- 示例中展示了模型加载、推理执行、后处理以及结果可视化的完整过程；
- 具体使用方法、参数说明及接口说明请参考 `runtime/python/README.md`；

---

## 模型评估（Evaluator）

`evaluator/` 用于模型精度、性能及数值一致性评估，相关内容暂未提供，待补充。

---

## 推理结果

![Inference Result](test_data/result.jpg)

---

## License

遵循 Model Zoo 顶层 License。
