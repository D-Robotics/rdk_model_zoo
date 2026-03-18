<div align="center">
  <p><b>⚠️ 注意：本仓库目前正处于迁移重构中，部分功能和文档暂未完全开发完成，敬请谅解。</b></p>
</div>

<div align="center">
  <img src="resource/imgs/model_zoo_logo.jpg" width="60%" alt="RDK Model Zoo Logo"/>
</div>

<div align="center">
  <h1 align="center">RDK Model Zoo</h1>
  <p align="center">
    <b>基于 D-Robotics BPU 的开箱即用 AI 模型部署 Pipeline 与全链路转换教程</b>
  </p>
</div>

<div align="center">

[English](./README.md) | **简体中文**

<p align="center">
  <a href="https://github.com/D-Robotics/rdk_model_zoo/stargazers"><img src="https://img.shields.io/github/stars/D-Robotics/rdk_model_zoo?style=flat-square&logo=github&color=blue" alt="Stars"></a>
  <a href="https://github.com/D-Robotics/rdk_model_zoo/network/members"><img src="https://img.shields.io/github/forks/D-Robotics/rdk_model_zoo?style=flat-square&logo=github&color=blue" alt="Forks"></a>
  <a href="https://github.com/D-Robotics/rdk_model_zoo/pulls"><img src="https://img.shields.io/badge/PRs-Welcome-brightgreen.svg?style=flat-square" alt="PRs Welcome"></a>
  <a href="https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x5/LICENSE"><img src="https://img.shields.io/github/license/D-Robotics/rdk_model_zoo?style=flat-square" alt="License"></a>
  <a href="https://developer.d-robotics.cc"><img src="https://img.shields.io/badge/Community-D--Robotics-orange.svg?style=flat-square" alt="Community"></a>
</p>

</div>

## 仓库简介 (Introduction)

> **使命**：致力于为地瓜机器人开发者提供极致性能、开箱即用、覆盖全场景的 AI 部署验证体验。

本仓库是 D-Robotics（地瓜机器人）官方提供的 BPU 模型示例与工具集合（Model Zoo），面向运行在 BPU（Brain Processing Unit）上的 AI 模型部署与应用开发，用于帮助开发者**快速上手 BPU**、**快速跑通模型推理流程**。

仓库中收录了覆盖多个 AI 领域的 BPU 可运行模型，并提供从 **原始模型 (PyTorch/ONNX) -> 定点量化转换 → 推理运行 → 结果解析 → 示例验证** 的完整参考实现，帮助用户以最小成本理解并使用 BPU 能力。

### 仓库核心价值
- 🚀 **快速把 BPU 用起来**：提供开箱即用的推理 Pipeline，帮助用户在最短时间内完成 BPU 推理验证及性能评估。
- 🧩 **完整端到端示例**：覆盖从算法导出、定点量化转换到板端高效运行 (`.bin` / `.hbm`) 的全过程。包含模型加载、前处理、BPU 推理执行、后处理与结果可视化。
- 📐 **规范化设计与接口文档**：提供统一的目录结构与示例代码规范，支持 Python (`hbm_runtime`) 与 C/C++ 接口，便于客户快速理解、二次开发，降低集成与维护成本。
- **🌐 全场景覆盖**：涵盖分类、检测、分割、姿态估计、OCR 及 LLM 等前沿多模态模型。

### 硬件与系统支持
- **RDK X5 (Bayse-e)**: 推荐使用 RDK OS >= 3.5.0 (基于 Ubuntu 22.04 aarch64, TROS-Humble)。
- **RDK S100/S600**: 请参考专用仓库 [RDK Model Zoo S](https://github.com/d-Robotics/rdk_model_zoo_s)。

---

## 仓库目录结构

本仓库采用分层清晰、职责明确的目录结构，以便用户快速定位所需内容并开始使用。

<details>
<summary><b>📂 点击展开查看项目目录架构</b></summary>

<br>

```bash
rdk_model_zoo/
├── demos/                 # 🚀 核心模型示例区 (按任务分类)
│   ├── classification/    # 分类网络 (MobileNet, ResNet, ConvNeXt...)
│   ├── detect/            # 检测网络 (YOLOv5~v12, FCOS...)
│   ├── Seg/               # 分割网络 (YOLO-Seg...)
│   ├── Pose/              # 姿态估计
│   ├── OCR/               # 文字识别 (PaddleOCR)
│   ├── llm/               # 大语言/多模态模型 (CLIP, YOLO-World)
│   └── tools/             # 批量性能测试与评测工具
├── docs/                  # 📖 规范指南与参考文档
├── datasets/              # 🗂️ 示例数据集及下载脚本
├── utils/                 # 🛠️ C++/Python 跨项目通用前后处理工具库
└── resource/              # 🖼️ 静态资源 (测试图片、Logo 等)
```
</details>

---

## 快速开始 (Quick Start)

本仓库中的模型均已按领域进行分类，并汇总在下方的 **模型支持矩阵** 中。用户可通过如下步骤快速运行目标模型：

1. **查找模型**：根据自身需求，在下方的模型列表中查找目标模型。
2. **连接硬件**：确保 RDK 板卡已通电联网。推荐使用 SSH (Windows Terminal / MobaXterm) 或 VSCode Remote SSH 插件连接板卡。
3. **安装依赖**：在 RDK 板卡终端执行命令安装基础库 (RDK OS >= 3.5.0 系统已内置，可跳过)：`pip install hbm_runtime`
4. **运行示例**：根据表格中提供的路径进入对应的模型目录，**仔细阅读该目录下的 `README.md`**，按照指引即可完成推理示例的运行与验证。

> **以 YOLO11 目标检测为例：**
> ```bash
> # 1. 克隆仓库
> git clone https://github.com/D-Robotics/rdk_model_zoo.git
> cd rdk_model_zoo
> 
> # 2. 进入模型目录并阅读该目录下的 README
> cd demos/detect/YOLO11/YOLO11-Detect_YUV420SP
> 
> # 3. 运行推理代码 (脚本将自动下载量化模型并执行推理)
> python3 YOLO11_Detect_YUV420SP.py --model-path ./model/yolo11n_det_640x640_nv12.bin --test-img ./data/bus.jpg
> ```

**推理结果展示:**
<div align="center">
  <img src="resource/imgs/demo_rdkx5_yolov10n_detect.jpg" width="80%" alt="Inference Result"/>
</div>

---

## 模型支持矩阵 (Model Zoo Matrix)

下表按 **任务类型** 对当前仓库中已提供的模型进行分类，方便快速查找与定位。每个模型的详细说明、使用方法和示例，请点击对应的 `Code` 链接进入目录查看。

| 任务类型 (Task) | 代表模型 (Models) | 示例代码 (Code) |
| :--- | :--- | :---: |
| **图像分类** | MobileNet (V1-V4), EfficientNet, ConvNeXt, ResNet, FastViT 等 20+ 模型 | [Code](./demos/classification) |
| **目标检测** | YOLOv5, YOLOv8, YOLOv10, YOLO11, YOLO12, FCOS, LPRNet | [Code](./demos/detect) |
| **实例分割** | YOLOv8-Seg, YOLO11-Seg, YOLOE-11-Seg-Prompt-Free | [Code](./demos/Seg) |
| **姿态估计** | YOLO11-Pose | [Code](./demos/Pose) |
| **大模型 (LLMs)** | CLIP, YOLO-World | [Code](./demos/llm) |
| **文字识别 (OCR)** | PaddleOCR | [Code](./demos/OCR) |
| **视觉专项** | MODNet (人像抠图) | [Code](./demos/Vision) |

*(持续更新中... 欢迎提交 PR 补充更多模型)*

---

## 文档说明与学习资源

为了帮助您更好地理解和使用 RDK 平台及本仓库代码，请参考以下文档：

- **模型说明**：每个模型的顶层目录的 `README.md` 中，都有该模型的整体介绍及运行指引，请直接到相关目录查看。
- **源码参考**：每个模型都有详细的接口介绍。如需了解代码层面的接口信息，请阅读 **[源码文档说明](docs/source_reference/README.md)**，根据介绍构建或浏览代码 API 文档。
- **开发规范**：如需二次开发或提交自己的模型 Sample，请仔细阅读 **[Model Zoo 仓库规范指南](./docs/Model_Zoo_Repository_Guidelines.md)**。
- **工具链手册**：
  - [RDK X5 算法工具链文档](https://developer.d-robotics.cc/api/v1/fileData/x5_doc-v126cn/index.html)
  - [RDK X3 算法工具链文档](https://developer.d-robotics.cc/api/v1/fileData/horizon_xj3_open_explorer_cn_doc/index.html)
- **开发者论坛**：[地瓜开发者社区](https://developer.d-robotics.cc/)
- **入门指南**: [RDK 用户手册](https://developer.d-robotics.cc/information)

---

## 常见问题解答 (FAQ)

<details>
<summary><b>1. 自己训练模型的精度不满足预期？</b></summary>
<br>

- 请检查 OpenExplorer 工具链 Docker、板端 `libdnn.so` 的版本是否均为目前发布的最新版本。
- 请检查在导出模型时，是否有按照对应示例文件夹内 README 的要求进行结构调整或算子替换。
- 检查量化验证阶段，每一个输出节点的余弦相似度是否均达到 0.999 以上 (保底 0.99)。
</details>

<details>
<summary><b>2. 自己训练模型的速度不满足预期？</b></summary>
<br>

- Python API 的推理性能会弱于 C/C++ API。如对极致性能有要求，请基于 C/C++ API 测试。
- 性能数据(纯前向)不包含前后处理，与完整 demo 的端到端耗时存在差异。通常采用 **NV12** 输入的模型可以做到极高吞吐量。
- 确认板卡是否已通过命令定频到最高频率。
- 检查是否有其他应用占用了 CPU/BPU 或 DDR 带宽资源。
</details>

<details>
<summary><b>3. 如何解决模型量化掉精度问题？</b></summary>
<br>

- 根据平台版本，先参考对应平台的文档中的 PTQ 精度 debug 章节进行分析。
- 如果是由于模型结构特性、权重分布导致 int8 量化掉精度严重，请考虑使用混合量化 (Mixed Precision) 或 QAT (量化感知训练)。
</details>

<details>
<summary><b>4. 报错 "Can't reshape 1354752 in (1,3,640,640)" 怎么解决？</b></summary>
<br>

请修改同级目录下 `preprocess.py` 文件中的分辨率设置，修改为准备转化的 onnx 对应的输入分辨率大小。同时删除之前生成的旧校准数据集，并重新运行校准数据生成脚本。
</details>

<details>
<summary><b>5. mAP 精度相比官方结果（如 ultralytics）低一些是正常的吗？</b></summary>
<br>

是的，这属于正常现象。主要原因包括：
- 官方测试通常使用动态 shape 和高精度浮点运算，而部署时使用了固定 shape 且经过了 INT8 量化。
- 我们使用 `pycocotools` 的评估脚本与官方评估脚本在细节实现上可能存在细微差异。
- NCHW-RGB888 输入格式转换为针对 BPU 优化的 YUV420SP (NV12) 输入格式时，会带来极少量的像素级精度损失。
</details>

<details>
<summary><b>6. 模型进行推理时会使用 CPU 处理吗？</b></summary>
<br>

会的。在模型转化的过程中，无法量化的算子、不满足 BPU 硬件约束的算子，或者不满足被动量化逻辑的算子，会**回退 (Fallback)** 到 CPU 计算。此外，对于纯 BPU 算子组成的 `.bin` 模型，输入/输出端通常会带有量化 (Float->Int) 和反量化 (Int->Float) 节点，这些转换工作是由 CPU 执行的。
</details>

---

## 社区与贡献 (Community & Contribution)

### Star 增长趋势
[![Star History Chart](https://api.star-history.com/svg?repos=D-Robotics/rdk_model_zoo&type=Date)](https://star-history.com/#D-Robotics/rdk_model_zoo&Date)

我们非常欢迎开发者参与共建 RDK Model Zoo！如果您有任何问题或建议，请随时在 [GitHub Issues](https://github.com/D-Robotics/rdk_model_zoo/issues) 提出，或在[地瓜开发者社区](https://developer.d-robotics.cc/)发帖交流。

## 许可证 (License)

本项目采用 [Apache License 2.0](./LICENSE) 开源协议。
