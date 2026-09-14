[English](./README.md) | 简体中文

# MODNet 模型说明

本目录给出 MODNet sample 在 Model Zoo 中的完整使用说明，包括算法概览、模型转换、运行时推理、模型文件管理和评测说明。

---

## 算法介绍（Algorithm Overview）

MODNet（Mobile-friendly One-stage Deep Image Matting Network）是一种轻量级深度学习人像抠图模型。它仅需单张 RGB 图像即可直接预测 alpha matte，无需 trimap 作为辅助输入。

- **论文**: [Is a Green Screen Really Necessary for Real-Time Portrait Matting?](https://arxiv.org/abs/2011.11961)
- **官方实现**: [ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet)

### 算法功能

MODNet 能完成以下任务：

- 单张 RGB 图像的人像抠图
- Alpha matte 预测（无需 trimap）

### 算法特性

- **单阶段端到端**：从 RGB 图像直接预测 alpha matte，无需 trimap
- **移动端友好**：专为资源受限设备的实时推理设计
- **语义-空间-细节融合**：采用三模块架构（语义估计、细节细化、细节注意力）实现高质量抠图

---

## 目录结构（Directory Structure）

本目录包含：

```bash
.
├── conversion                          # 模型转换流程
│   ├── onnx_export                     # ONNX 导出脚本
│   ├── ptq_yamls                       # PTQ 配置 YAML 文件
│   ├── README.md                       # 使用说明 (英文)
│   └── README_cn.md                    # 使用说明 (中文)
├── evaluator                           # 模型评估相关内容
│   ├── README.md                       # 使用说明 (英文)
│   └── README_cn.md                    # 使用说明 (中文)
├── model                               # 模型文件及下载脚本
│   ├── download_model.sh               # BIN 模型下载脚本
│   ├── README.md                       # 使用说明 (英文)
│   └── README_cn.md                    # 使用说明 (中文)
├── runtime                             # 模型推理示例
│   └── python                          # Python 推理示例
│       ├── main.py                     # Python 推理入口脚本
│       ├── modnet.py                   # MODNet 推理与后处理实现
│       ├── run.sh                      # Python 示例运行脚本
│       ├── README.md                   # 使用说明 (英文)
│       └── README_cn.md                # 使用说明 (中文)
├── test_data                           # 推理结果与示例数据
│   ├── person.jpg                      # 示例输入图像
│   └── bg.jpg                          # 示例背景图像
└── README.md                           # MODNet 示例整体说明与快速指引
```

---

## 快速体验（QuickStart）

为了便于用户快速上手体验，每个模型均提供了 `run.sh` 脚本，用户运行此脚本即可一键运行相应模型。

### Python

- 进入 `runtime` 目录下的 `python` 目录，运行 `run.sh` 脚本，即可快速体验
    ```bash
    cd runtime/python/
    chmod +x run.sh
    ./run.sh
    ```
- 若想了解 `python` 代码的详细使用方法，请参考 [runtime/python/README_cn.md](./runtime/python/README_cn.md)

---

## 模型转换（Model Conversion）

- ModelZoo 已提供适配完成的 BIN 模型文件，用户可直接运行 `model` 目录下的 `download_model.sh` 脚本下载并使用，如不关心模型转换流程，**可跳过本小节**。

- 如需自定义模型转换参数，或了解完整的模型转换流程，请参考 [conversion/README_cn.md](./conversion/README_cn.md)。

---

## 模型推理（Runtime）

MODNet 模型推理示例提供 Python 实现方式。

### Python 版本

- 以脚本形式提供，适合快速验证模型效果与算法流程
- 示例中展示了模型加载、推理执行、后处理以及结果合成的完整过程
- 具体使用方法、参数说明及接口说明请参考 [runtime/python/README_cn.md](./runtime/python/README_cn.md)

---

## 模型评估（Evaluator）

`evaluator/` 用于模型精度、性能及数值一致性评估，详细说明请参考 [evaluator/README_cn.md](./evaluator/README_cn.md)。

---

## 性能数据

下表展示 MODNet 模型在 RDK X5 平台上的实际测试性能数据。

| 模型 | 尺寸 | 输入格式 | 延迟 (ms) | 帧率 (FPS) |
| --- | --- | --- | --- | --- |
| MODNet | 512x512 | Float32 NCHW RGB | 89.88 | 11.12 |
| MODNet (2 线程) | 512x512 | Float32 NCHW RGB | 130.49 | 15.27 |

**说明：**
1. 测试平台：RDK X5，CPU 8xA55@1.8G，BPU 1xBayes-e@1G (10TOPS INT8)
2. 单线程延迟数据为单帧单线程、单 BPU 核心下的理想情况
3. 多线程帧率数据为 2 线程并发场景

---

## License

遵循 Model Zoo 顶层 License。
