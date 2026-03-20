# YOLOe11-Seg 模型说明

本目录描述 YOLOe11-Seg 在本 Model Zoo 中的完整使用流程，包括：算法介绍、模型转换、运行时推理（Python/C++）、前后处理接口说明，以及模型评估步骤。

> **平台限制：本模型不支持 S600 平台。** Python 脚本和 C++ 程序均包含平台检查，在 S600 上运行将直接报错退出。

---

## 算法介绍（Algorithm Overview）

YOLOe11-Seg 是基于 Ultralytics YOLO11 系列的开放词表（Open Vocabulary）实例分割模型，支持 **4585 个类别**，在标准实例分割能力之上扩展了开放词表识别能力，具有以下特性：

- **开放词表（Open Vocabulary）检测**：支持 4585 个类别，超越传统 COCO 80 类限制
- **无锚点（Anchor-Free）检测**：基于 DFL 分布回归精确定位边界框
- **原型掩码（Prototype-based Mask）**：通过原型特征与 MCES 系数线性组合生成实例掩码
- **高效推理**：C++ 实现中使用 OpenMP 并行化加速 4585 类 argmax 计算
- **适合端侧部署**：在 RDK S100 等嵌入式平台上高效运行

### 算法功能

YOLOe11-Seg 能完成以下任务：

- 开放词表多类别目标检测与实例分割（4585 类）
- 输出类别、置信度、边界框及逐像素实例掩码

### 原始资料

- YOLO11 官方仓库：https://github.com/ultralytics/ultralytics

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
|   |   |   `-- yoloe11seg.hpp              # YOLO11ESeg 模型封装接口
|   |   |-- src                             # C++ 源码
|   |   |   |-- main.cpp                    # 推理入口程序
|   |   |   `-- yoloe11seg.cpp              # YOLOe11-Seg 推理实现（含 OpenMP 并行）
|   |   |-- CMakeLists.txt                  # CMake 构建配置
|   |   |-- README.md                       # C++ 推理示例使用说明
|   |   `-- run.sh                          # C++ 示例运行脚本
|   `-- python                              # Python 推理示例
|       |-- README.md                       # Python 推理示例使用说明
|       |-- main.py                         # Python 推理入口脚本
|       |-- run.sh                          # Python 示例运行脚本
|       `-- yoloe11seg.py                   # YOLOe11-Seg 推理与后处理实现
|-- test_data                               # 测试数据与推理结果
|   |-- office_desk.jpg                     # 示例测试图片
|   |-- coco_extended.names                 # 开放词表类别标签文件（4585 类）
|   `-- result.jpg                          # 示例推理结果图像
`-- README.md                               # YOLOe11-Seg 示例整体说明与快速指引
```

---

## 快速体验（QuickStart）

为了便于用户快速上手体验，每个模型均提供了 `run.sh` 脚本，用户运行此脚本即可一键运行相应模型，此脚本主要进行如下操作：
- 检测平台兼容性（S600 不支持，直接退出）；
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

- 如需自定义模型转换参数，或了解完整的模型转换流程，请参考 `conversion/README.md`。

---

## 模型推理（Runtime）

YOLOe11-Seg 模型推理示例同时提供 C++ 和 Python 两种实现方式。

### C++ 版本

- 提供完整的工程化示例，适合集成到实际应用中；
- C++ 实现使用 OpenMP 并行化加速 4585 类 argmax 计算；
- 具体编译、运行方式及接口说明请参考 `runtime/cpp/README.md`；

### Python 版本

- 以脚本形式提供，适合快速验证模型效果与算法流程；
- 具体使用方法、参数说明及接口说明请参考 `runtime/python/README.md`；

---

## 模型评估（Evaluator）

`evaluator/` 用于模型精度、性能及数值一致性评估，详细说明请参考该目录。

---

## 推理结果

![Inference Result](test_data/result.jpg)

---

## License

遵循 Model Zoo 顶层 License。
