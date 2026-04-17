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

仓库中收录了覆盖多个 AI 领域的 BPU 可运行模型，并提供从 **原始模型 (PyTorch/ONNX) -> 定点量化转换 -> 推理运行 -> 结果解析 -> 示例验证** 的完整参考实现，帮助用户以最小成本理解并使用 BPU 能力。

### 仓库核心价值
- 🚀 **快速把 BPU 用起来**：提供开箱即用的推理 Pipeline，帮助用户在最短时间内完成 BPU 推理验证及性能评估。
- 🧩 **完整端到端示例**：覆盖从算法导出、定点量化转换到板端高效运行（`.bin` / `.hbm`）的全过程。包含模型加载、前处理、BPU 推理执行、后处理与结果可视化。
- 📐 **规范化设计与接口文档**：提供统一的目录结构与示例代码规范，支持 Python（`hbm_runtime`）与 C/C++ 接口，便于快速理解和二次开发，降低集成与维护成本。
- 🌐 **全场景覆盖**：涵盖分类、检测、分割、姿态估计、OCR 以及 LLM 等前沿多模态模型。

### 硬件与系统支持
- **RDK X5 (Bayse-e)**：推荐使用 RDK OS >= 3.5.0（基于 Ubuntu 22.04 aarch64, TROS-Humble）。
- **RDK S100/S600**：请参考专用仓库 [RDK Model Zoo S](https://github.com/d-Robotics/rdk_model_zoo_s)。

---

## 仓库目录结构

本仓库将**已完成标准化重构的交付内容**统一放在 `samples/`，将**仍处于旧结构或尚未完成标准化重构的内容**保留在 `demos/`。

<details>
<summary><b>点击展开项目目录结构</b></summary>

<br>

```bash
rdk_model_zoo/
├── samples/               # 已完成标准化重构的交付样例
│   └── vision/
│       ├── convnext/
│       ├── fcos/
│       ├── PaddleOCR/
│       ├── ultralytics_yolo/
│       ├── ultralytics_yolo26/
│       └── yolov5/
├── demos/                 # 尚未完成标准化重构的旧版或过渡内容
│   ├── classification/    # 分类模型集合（20+ 模型）
│   ├── detect/            # LPRNet 等检测类 demo
│   ├── Seg/               # 分割类旧版 demo
│   ├── Vision/            # MODNet 等视觉类 demo
│   ├── llm/               # LLM / 多模态 demo
│   └── solutions/         # 端到端解决方案 demo
├── docs/                  # 项目规范与参考文档
├── datasets/              # 数据集与下载脚本
├── utils/                 # 公共 C++ / Python 工具
└── resource/              # 静态资源（图片、Logo 等）
```
</details>

---

## 快速开始 (Quick Start)

优先使用 `samples/` 中的标准化样例。只有在目标模型还未完成标准化重构时，才进入 `demos/`。

1. **选择正确目录**
   - 标准化样例：`samples/vision/...`
   - 旧版 / 待重构内容：`demos/...`
2. **检查系统版本**
   - 确保目标板卡系统版本满足 `RDK OS >= 3.5.0`。
3. **连接硬件**
   - 确保 RDK 板卡上电并可通过 SSH 或 VSCode Remote SSH 访问。
4. **先阅读对应 README**
   - 进入目标目录后先阅读 `README.md` / `README_cn.md`，再执行命令。

> **示例 A：运行标准化 Sample（`samples/`）**
> ```bash
> # 进入标准化 Sample 目录
> cd samples/vision/yolov5/runtime/python
>
> # 运行推理（脚本会自动下载默认模型）
> bash run.sh
> ```

> **示例 B：查看旧版 / 待重构 Demo（`demos/`）**
> ```bash
> # 进入旧版 demo 目录
> cd demos/detect/LPRNet
>
> # 阅读该 demo 的 README 并按说明运行
> ```

**推理结果示例：**
<div align="center">
  <img src="resource/imgs/demo_rdkx5_yolov10n_detect.jpg" width="80%" alt="Inference Result"/>
</div>

---

## 模型支持矩阵 (Model Zoo Matrix)

### 标准化 Samples

以下目录已经完成标准化重构，是当前推荐的使用入口。

| 类别 | 模型 | 路径 |
| :--- | :--- | :---: |
| **分类** | ConvNeXt | [Code](./samples/vision/convnext) |
| **检测** | FCOS | [Code](./samples/vision/fcos) |
| **检测** | YOLOv5 | [Code](./samples/vision/yolov5) |
| **Ultralytics YOLO** | YOLOv5u、YOLOv8、YOLOv9、YOLOv10、YOLO11、YOLO12、YOLO13、YOLO26 | [Code](./samples/vision/ultralytics_yolo)、[YOLO26](./samples/vision/ultralytics_yolo26) |
| **OCR** | PaddleOCR | [Code](./samples/vision/PaddleOCR) |

### Legacy / 待重构 Demos

以下目录已回迁到 `demos/`，作为旧版内容或待后续标准化重构的内容保留。

| 类别 | 代表模型 | 路径 |
| :--- | :--- | :---: |
| **分类** | MobileNet（V1-V4）、EfficientNet、ResNet、RepViT、FastViT 等分类模型 | [Code](./demos/classification) |
| **检测** | LPRNet | [Code](./demos/detect) |
| **分割** | YOLOE-11-Seg-Prompt-Free | [Code](./demos/Seg) |
| **视觉特化** | MODNet | [Code](./demos/Vision) |
| **大模型 / 多模态** | CLIP、YOLO-World | [Code](./demos/llm) |
| **解决方案** | RDK LLM Solutions、RDK Video Solutions | [Code](./demos/solutions) |

---

## 文档说明与学习资源

为了帮助你更好地理解和使用 RDK 平台与本仓库代码，建议优先阅读以下文档：

- **模型说明**
  - 每个模型目录下的 `README.md` / `README_cn.md` 都包含整体介绍、运行方法和目录说明。
- **源码参考**
  - 如需了解代码级接口说明，请参考 **[源码文档说明](./docs/source_reference/README.md)**。
- **开发规范**
  - 如需新增或重构 Sample，请先阅读 **[Model Zoo 仓库规范指南](./docs/Model_Zoo_Repository_Guidelines.md)**。
- **工具链文档**
  - [RDK X5 算法工具链文档](https://developer.d-robotics.cc/api/v1/fileData/x5_doc-v126cn/index.html)
  - [RDK X3 算法工具链文档](https://developer.d-robotics.cc/api/v1/fileData/horizon_xj3_open_explorer_cn_doc/index.html)
- **开发者社区**
  - [D-Robotics 开发者社区](https://developer.d-robotics.cc/)
- **用户手册**
  - [RDK 用户手册](https://developer.d-robotics.cc/information)

---

## 常见问题解答 (FAQ)

<details>
<summary><b>1. 自己训练模型的精度不满足预期？</b></summary>
<br>

- 检查 OpenExplorer Docker 与板端 `libdnn.so` 是否为当前推荐版本。
- 检查模型导出时是否按对应示例 README 的要求完成结构调整或算子替换。
- 检查量化验证阶段各输出节点的余弦相似度是否达到 0.999 以上（最低不低于 0.99）。
</details>

<details>
<summary><b>2. 自己训练模型的速度不满足预期？</b></summary>
<br>

- Python API 的性能通常低于 C/C++，如需极限性能请优先使用 C/C++。
- Benchmark 数据通常只统计纯前向，不包含前后处理，完整 demo 端到端耗时会更高。
- 使用 **NV12** 输入的模型通常更容易获得最高 BPU 吞吐。
- 请确认板卡 CPU / BPU 已设置为高性能模式，并避免其他进程抢占资源。
</details>

<details>
<summary><b>3. 如何解决模型量化掉精度问题？</b></summary>
<br>

- 请优先参考对应平台工具链文档中的 PTQ 精度调试章节。
- 若模型结构本身对 INT8 敏感，可考虑 Mixed Precision 或 QAT（量化感知训练）。
</details>

<details>
<summary><b>4. 报错 "Can't reshape 1354752 in (1,3,640,640)" 怎么解决？</b></summary>
<br>

请修改同目录下 `preprocess.py` 中的分辨率设置，使其与待转换 ONNX 模型的输入尺寸一致。同时删除旧的校准数据并重新生成。
</details>

<details>
<summary><b>5. mAP 精度相比官方结果（如 Ultralytics）偏低是否正常？</b></summary>
<br>

一般属于正常现象，常见原因包括：
- 官方测试通常使用动态 shape 和浮点精度，而部署版本使用固定 shape 与 INT8 量化。
- `pycocotools` 评测脚本和官方评测实现之间可能存在细微差异。
- 从 RGB 输入转换为 NV12 输入时会带来少量像素级误差。
</details>

<details>
<summary><b>6. 模型推理时会使用 CPU 吗？</b></summary>
<br>

会。无法量化的算子、无法映射到 BPU 的算子，或量化 / 反量化节点都会由 CPU 执行。即使是以 BPU 为主的 `.bin` 模型，输入输出端通常也会包含 CPU 参与的转换过程。
</details>

---

## 社区与贡献 (Community & Contribution)

### Star 增长趋势
[![Star History Chart](https://api.star-history.com/svg?repos=D-Robotics/rdk_model_zoo&type=Date)](https://star-history.com/#D-Robotics/rdk_model_zoo&Date)

欢迎参与共建 RDK Model Zoo。如有问题或建议，请通过 [GitHub Issues](https://github.com/D-Robotics/rdk_model_zoo/issues) 提出，或在 [D-Robotics 开发者社区](https://developer.d-robotics.cc/) 交流。

## 许可证 (License)

本项目采用 [Apache License 2.0](./LICENSE) 开源协议。
