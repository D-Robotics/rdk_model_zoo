# ByteTrack 模型说明

本目录描述 ByteTrack 在本 Model Zoo 中的完整使用流程，包括：算法介绍、模型转换、运行时推理（Python）、可复用的前后处理接口说明，以及模型评估步骤。

---

## 算法介绍（Algorithm Overview）

ByteTrack 是一种简单、快速且强大的多目标跟踪算法。其核心创新在于关联方法 BYTE，通过关联每个检测框（不仅仅是高分框）来处理遮挡和低分检测。本示例演示了如何在 RDK 平台上结合 YOLOv5 检测模型和 ByteTrack 算法进行实时多目标跟踪。

### 算法功能
ByteTrack 能完成以下任务：
- 多目标实时跟踪
- 输出目标轨迹 ID 和边界框

### 原始资料
- 论文: [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)
- 官方仓库: [ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack)

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
|   |-- download_model.sh               # HBM 模型下载脚本
|   `-- README.md                       # HBM 模型下载使用说明
|-- runtime                             # 模型推理示例
|   `-- python                          # Python 推理示例
|       |-- tracker                     # ByteTrack 核心算法实现
|       |   |-- byte_tracker.py         # 跟踪器主逻辑
|       |   |-- kalman_filter.py        # 卡尔曼滤波实现
|       |   `-- ...                     # 其他辅助文件
|       |-- README.md                   # Python 推理示例使用说明
|       |-- main.py                     # Python 推理入口脚本
|       |-- run.sh                      # Python 示例运行脚本
|       |-- bytetrack.py                # ByteTrack 模型封装
|       `-- yolov5.py                   # YOLOv5 检测器封装
|-- test_data                           # 推理结果
|   `-- result.mp4                      # 示例推理结果视频
`-- README.md                           # ByteTrack 示例整体说明与快速指引
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
- 若想了解`python`代码的详细使用方法，或step by step运行模型请参考`runtime/python/README.md`；

---

## 模型转换（Model Conversion）

- ModelZoo 已提供适配完成的 HBM 模型文件，用户可直接运行`model` 目录下的`download_model.sh`脚本下载并使用，如不关心模型转换流程，**可跳过本小节**。

- 如需自定义模型转换参数，或了解完整的模型转换流程，请参考`conversion/README.md`。

---

## 模型推理（Runtime）

ByteTrack 模型推理示例目前仅提供 Python 实现方式。

### C++ 版本

    - 暂未提供。

### Python 版本

    - 以脚本形式提供，适合快速验证模型效果与算法流程;

    - 示例中展示了模型加载、推理执行、后处理以及结果可视化的完整过程;

    - 具体使用方法、参数说明及接口说明请参考 `runtime/python/README.md`;

---

## 模型评估（Evaluator）

`evaluator/` 用于模型精度、性能及数值一致性评估，详细说明请参考该目录。

---

## 推理结果

![Inference Result](test_data/result.jpg)

---

## License
遵循 Model Zoo 顶层 License。
