[English](./README.md) | 简体中文

# KWS 模型说明

本目录描述 KWS（关键词检测）在本 Model Zoo 中的完整使用流程，包括：算法介绍、模型转换、运行时推理（Python）、可复用的前后处理接口说明，以及模型评估步骤。

> 本模型支持 **RDK S100** 平台。

---

## 算法介绍

KWS（Keyword Spotting）是基于深度学习的唤醒词检测模型，采用 MDTC（Multi-Scale Dynamic Temporal Convolution）算法，具有以下特性：

- **多尺度卷积**：捕获不同时间尺度的语音特征，提升检测鲁棒性
- **动态卷积**：自适应调整卷积权重，适应不同说话人和环境
- **边缘友好**：轻量化设计，适合嵌入式平台部署
- **高精度**：在 BPU 上高效运行，检测准确率高

### 算法功能

KWS 能完成以下任务：

- 关键词检测（输入 .wav 音频，输出关键词置信度分数）

### 原始资料

- 框架: PaddlePaddle + PaddleAudio
- 算法: MDTC (Multi-Scale Dynamic Temporal Convolution)

---

## 平台兼容性

| 平台       | 是否支持 | 说明                            |
|-----------|---------|-------------------------------|
| RDK S100  | ✅ 支持  | 模型已针对 S100 BPU 编译，推荐使用 |
| RDK S600  | ❌ 不支持 | 暂未适配                       |

---

## 目录结构

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
|       |-- README.md                   # Python 推理示例使用说明
|       |-- main.py                     # Python 推理入口脚本
|       |-- kws.py                      # KWS 模型封装
|       `-- run.sh                      # Python 示例运行脚本
|-- test_data                           # 测试数据
|   `-- sample.wav                      # 示例唤醒词音频（"hey snips"）
`-- README.md                           # KWS 示例整体说明与快速指引
```

---

## 快速体验

每个模型提供了 `run.sh` 脚本，运行此脚本即可一键运行相应模型：
- 检测系统环境是否满足要求，若不满足则自动安装相应包；
- 检测推理所需的 hbm 模型文件是否存在，不存在则自动下载；
- 运行相应的 Python 脚本；

### Python

- 进入 `runtime/python/` 目录，运行 `run.sh` 脚本
    ```bash
    cd runtime/python/
    ./run.sh
    ```
- 若想了解 Python 代码的详细使用方法，或 step by step 运行模型请参考 `runtime/python/README.md`；

---

## 模型转换

- ModelZoo 已提供适配完成的 HBM 模型文件，用户可直接运行 `model` 目录下的 `download_model.sh` 脚本下载并使用，如不关心模型转换流程，**可跳过本小节**。
- 如需自定义模型转换参数，或了解完整的模型转换流程，请参考 `conversion/README.md`。

---

## 模型推理

KWS 模型推理示例目前仅提供 Python 实现方式。

### Python 版本

- 以脚本形式提供，适合快速验证模型效果与算法流程；
- 示例中展示了模型加载、音频预处理（fbank 特征提取）、推理执行以及置信度分数输出的完整过程；
- 具体使用方法、参数说明及接口说明请参考 `runtime/python/README.md`；

---

## 推理结果

运行成功后，置信度分数将打印到终端，示例输出如下：

```text
Keyword confidence score: 0.9850
```

---

## 模型评估

`evaluator/` 用于模型精度、性能及数值一致性评估，详细说明请参考该目录。

---

## License

遵循 Model Zoo 顶层 License。
