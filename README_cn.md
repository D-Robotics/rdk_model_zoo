<div align="center">
  <img src="docs/assets/model_zoo_logo.jpg" width="60%" alt="RDK Model Zoo Logo"/>
</div>

<div align="center">
  <h1 align="center">RDK Model Zoo — RDK S 系列</h1>
  <p align="center">
    <b>基于 D-Robotics BPU 的开箱即用 AI 模型部署流水线与全链路转换教程</b>
  </p>
</div>

<div align="center">

[English](./README.md) | **简体中文**

<p align="center">
  <a href="https://github.com/D-Robotics/rdk_model_zoo/stargazers"><img src="https://img.shields.io/github/stars/D-Robotics/rdk_model_zoo?style=flat-square&logo=github&color=blue" alt="Stars"></a>
  <a href="https://github.com/D-Robotics/rdk_model_zoo/network/members"><img src="https://img.shields.io/github/forks/D-Robotics/rdk_model_zoo?style=flat-square&logo=github&color=blue" alt="Forks"></a>
  <a href="https://github.com/D-Robotics/rdk_model_zoo/pulls"><img src="https://img.shields.io/badge/PRs-Welcome-brightgreen.svg?style=flat-square" alt="PRs Welcome"></a>
  <a href="https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_s/LICENSE"><img src="https://img.shields.io/github/license/D-Robotics/rdk_model_zoo?style=flat-square" alt="License"></a>
  <a href="https://developer.d-robotics.cc"><img src="https://img.shields.io/badge/Community-D--Robotics-orange.svg?style=flat-square" alt="Community"></a>
</p>

</div>

## 简介

> **使命**：致力于为 D-Robotics 开发者提供极致性能、开箱即用、全场景覆盖的 AI 部署验证体验。

本仓库是 D-Robotics（地瓜机器人）官方提供的 BPU 模型示例与工具集合（Model Zoo），面向运行在 BPU（Brain Processing Unit）上的 AI 模型部署与应用开发，帮助开发者**快速上手 BPU**、**快速跑通模型推理流程**。

仓库中收录了覆盖多个 AI 领域的 BPU 可运行模型，并提供从**原始模型（PyTorch/ONNX）→ 定点量化 → 推理运行 → 结果解析 → 示例验证**的完整参考实现，帮助用户以最小成本理解并使用 BPU 能力。

### 核心价值

- 🚀 **快速把 BPU 用起来**：提供开箱即用的推理流水线，帮助用户在最短时间内完成 BPU 推理验证与性能评测。
- 🧩 **完整端到端示例**：覆盖算法导出、定点量化到高效上板运行（`.hbm`）的全流程，包含模型加载、前处理、BPU 推理执行、后处理与结果可视化。
- 📐 **规范化设计与完整接口文档**：提供统一的目录结构与示例代码规范，支持 Python（`hbm_runtime`）与 C/C++ 双语言接口，便于理解、二次开发，降低集成与维护成本。
- 🌐 **全场景覆盖**：覆盖分类、检测、分割、姿态估计、深度估计、OCR、语音及多模态模型。

### 硬件与分支说明

本仓库使用硬件专属分支，清晰区分维护中的 sample、历史 demo 和板端专属文档。当前 `rdk_s` 分支是 RDK S 系列板卡（S100 / S100P / S600）的主要交付分支。

| 目标硬件 | 分支 | 说明 |
| :--- | :--- | :--- |
| RDK S 系列 | [`rdk_s`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_s) | **当前分支。** RDK S100、S100P、S600 的主要交付分支。 |
| RDK X5 | [`rdk_x5`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x5) | RDK X5 的主要交付分支。 |
| RDK X3 | [`rdk_x3`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x3) | RDK X3 设备专用分支。 |
| RDK S 历史 demo | [RDK Model Zoo S](https://github.com/D-Robotics/rdk_model_zoo_s) | RDK S 系列历史归档 demo 仓库。 |

---

## 目录结构

<details>
<summary><b>点击展开项目目录架构</b></summary>

<br>

```bash
rdk_model_zoo/                       # rdk_s 分支
|-- samples/
|   |-- vision/
|   |   |-- ultralytics_yolo/        # 检测 / 分割 / 姿态 / 分类
|   |   |-- ultralytics_yolo26/      # 检测 / 分割 / 姿态 / OBB / 分类
|   |   |-- yolov5/                  # 目标检测
|   |   |-- yolo11/                  # 目标检测
|   |   |-- yolo11_seg/              # 实例分割
|   |   |-- yolo11_pose/             # 姿态估计
|   |   |-- yoloe11_seg/             # 实例分割（无提示词）
|   |   |-- yolov13_imoonlab/        # 目标检测
|   |   |-- bytetrack/               # 多目标追踪
|   |   |-- resnet18/                # 图像分类
|   |   |-- resnet50/                # 图像分类
|   |   |-- resnet152/               # 图像分类
|   |   |-- mobilenetv1/             # 图像分类
|   |   |-- mobilenetv2/             # 图像分类
|   |   |-- mobilenetv3/             # 图像分类
|   |   |-- mobilenetv4/             # 图像分类
|   |   |-- efficientnet/            # 图像分类
|   |   |-- vit/                     # 图像分类
|   |   |-- 3dresnet/                # 视频动作分类
|   |   |-- unetmobilenet/           # 语义分割
|   |   |-- depth_anything_v2/       # 单目深度估计
|   |   |-- siglip/                  # VLM / VLA 视觉编码器
|   |   |-- pointnet/                # 点云零件分割
|   |   |-- lanenet/                 # 车道线检测
|   |   `-- paddle_ocr/             # OCR 文字检测与识别
|   |-- speech/
|   |   |-- asr/                     # 自动语音识别
|   |   `-- kws/                    # 关键词唤醒
|   `-- vla/
|       `-- act/                    # Action Chunking Transformer（机器人策略）
|-- docs/                            # 项目规范与参考文档
|-- datasets/                        # 示例数据集与下载脚本
|-- tros/                            # TROS 集成指南与示例
|-- utils/                           # 共享 Python 工具库
```

</details>

---

## 快速开始

1. **检查系统版本**：确认目标板卡运行支持的 RDK OS 版本。
2. **连接硬件**：确保 RDK S 板卡已上电并连接网络，推荐使用 SSH 或 VSCode Remote SSH。
3. **先阅读模型 README**：运行前务必打开目标目录的 `README.md`。
4. **运行示例**（以 YOLOv5 在 RDK S100 为例）：

```bash
cd samples/vision/yolov5/runtime/python
bash run.sh
```

`run.sh` 会自动下载模型、安装依赖并执行推理，输出图片保存在当前目录。

---

## 模型列表

| 类别 | 模型名称 | 模型路径 | 支持平台 | 详情 |
| :--- | :--- | :--- | :--- | :---: |
| 视觉多任务 | Ultralytics YOLO（YOLOv5u / YOLOv8 / YOLOv9 / YOLOv10 / YOLO11 / YOLO12） | `samples/vision/ultralytics_yolo` | S100 / S100P / S600 | [详情](./samples/vision/ultralytics_yolo) |
| 视觉多任务 | YOLO26 | `samples/vision/ultralytics_yolo26` | S100 / S100P / S600 | [详情](./samples/vision/ultralytics_yolo26) |
| 目标检测 | YOLOv5x | `samples/vision/yolov5` | S100 / S600 | [详情](./samples/vision/yolov5) |
| 目标检测 | YOLO11 | `samples/vision/yolo11` | S100 / S600 | [详情](./samples/vision/yolo11) |
| 目标检测 | YOLOv13（iMoonLab） | `samples/vision/yolov13_imoonlab` | S100 | [详情](./samples/vision/yolov13_imoonlab) |
| 多目标追踪 | ByteTrack | `samples/vision/bytetrack` | S100 / S100P / S600 | [详情](./samples/vision/bytetrack) |
| 实例分割 | YOLO11-Seg | `samples/vision/yolo11_seg` | S100 / S600 | [详情](./samples/vision/yolo11_seg) |
| 实例分割 | YOLOe11-Seg（无提示词） | `samples/vision/yoloe11_seg` | S100 | [详情](./samples/vision/yoloe11_seg) |
| 姿态估计 | YOLO11-Pose | `samples/vision/yolo11_pose` | S100 / S600 | [详情](./samples/vision/yolo11_pose) |
| 图像分类 | ResNet18 | `samples/vision/resnet18` | S100 / S600 | [详情](./samples/vision/resnet18) |
| 图像分类 | ResNet50 | `samples/vision/resnet50` | S100 / S600 | [详情](./samples/vision/resnet50) |
| 图像分类 | ResNet152 | `samples/vision/resnet152` | S100 / S600 | [详情](./samples/vision/resnet152) |
| 图像分类 | MobileNetV1 | `samples/vision/mobilenetv1` | S100 | [详情](./samples/vision/mobilenetv1) |
| 图像分类 | MobileNetV2 | `samples/vision/mobilenetv2` | S100 / S600 | [详情](./samples/vision/mobilenetv2) |
| 图像分类 | MobileNetV3 | `samples/vision/mobilenetv3` | S100 | [详情](./samples/vision/mobilenetv3) |
| 图像分类 | MobileNetV4 | `samples/vision/mobilenetv4` | S100 | [详情](./samples/vision/mobilenetv4) |
| 图像分类 | EfficientNet-Lite | `samples/vision/efficientnet` | S100 | [详情](./samples/vision/efficientnet) |
| 图像分类 | ViT | `samples/vision/vit` | S100 | [详情](./samples/vision/vit) |
| 图像分类 | 3D ResNet（视频动作分类） | `samples/vision/3dresnet` | S100 | [详情](./samples/vision/3dresnet) |
| 语义分割 | UnetMobileNet | `samples/vision/unetmobilenet` | S100 / S600 | [详情](./samples/vision/unetmobilenet) |
| 单目深度估计 | Depth Anything V2 | `samples/vision/depth_anything_v2` | S100 | [详情](./samples/vision/depth_anything_v2) |
| 视觉编码器 | SigLIP | `samples/vision/siglip` | S100 / S100P | [详情](./samples/vision/siglip) |
| 点云分割 | PointNet | `samples/vision/pointnet` | S100 | [详情](./samples/vision/pointnet) |
| 车道线检测 | LaneNet | `samples/vision/lanenet` | S100 | [详情](./samples/vision/lanenet) |
| 文字识别 | PaddleOCR | `samples/vision/paddle_ocr` | S100 | [详情](./samples/vision/paddle_ocr) |
| 语音识别 | ASR（Wav2Vec2） | `samples/speech/asr` | S100 / S600 | [详情](./samples/speech/asr) |
| 关键词唤醒 | KWS（MDTC） | `samples/speech/kws` | S100 | [详情](./samples/speech/kws) |
| 具身智能 / 机器人策略 | ACT（Action Chunking Transformer） | `samples/vla/act` | S100 / S600 | [详情](https://github.com/D-Robotics/rdk_LeRobot_tools) |

---

## 文档与资源

- **模型文档**：每个模型的顶层 `README.md` 提供整体介绍与运行指引。
- **源码参考**：代码层面的接口信息，请参阅 **[源码文档说明](./docs/source_reference/README.md)**。
- **仓库规范**：贡献或开发前，请仔细阅读 **[Model Zoo 仓库规范](./docs/Model_Zoo_Repository_Guidelines.md)**。
- **BPU Python API**：`hbm_runtime` 使用方式，请参阅 **[Python API 用户手册](./docs/Python_API_User_Guide.md)**。
- **UCP 接口**：`libdnn` / `libucp` 接口说明，请参阅 **[UCP 用户手册](./docs/UCP_User_Guide.md)**。
- **工具链手册**：[RDK S 系列 OE 工具链文档](https://developer.d-robotics.cc/rdk_s_doc/Advanced_development/toolchain_development/overview)
- **开发者社区**：[D-Robotics 开发者社区](https://developer.d-robotics.cc/)

---

## 常见问题

<details>
<summary><b>1. 模型精度不符合预期？</b></summary>
<br>

- 确认 OpenExplorer Docker 与板端 `hbm_runtime` 版本为最新。
- 检查模型导出是否按照对应 `conversion/README.md` 中的算子替换步骤操作。
- 确认量化验证阶段各输出节点余弦相似度 >= 0.999（最低 0.99）。
- 对于含 Transformer 结构的模型（如 ViT、Depth Anything V2、SigLIP），建议使用 int16 量化代替 int8。
</details>

<details>
<summary><b>2. 推理速度不符合预期？</b></summary>
<br>

- Python API 性能低于 C/C++，追求极致性能请使用 C/C++ 运行时。
- 性能 Benchmark 数据（纯前向）不含前后处理。**NV12** 输入模型通常能达到最高 BPU 吞吐量。
- 确保 CPU/BPU 频率已锁定为最高性能模式：

```bash
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
sudo bash -c "echo performance > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor"
sudo bash -c "echo performance > /sys/devices/system/bpu/bpu0/devfreq/28108000.bpu/governor"
```
</details>

<details>
<summary><b>3. 如何解决量化精度损失？</b></summary>
<br>

- 参考 OE 工具链文档中的 PTQ 精度调优章节。
- 若 INT8 损失较大（如含大量 Softmax 的 Transformer 模型），可在 YAML 配置中通过 `set_all_nodes_int16` 切换 INT16 量化。
- 严重情况下可考虑混合精度或 QAT（量化感知训练）。
</details>

<details>
<summary><b>4. 推理时模型会用 CPU 吗？</b></summary>
<br>

会。不可量化或 BPU 不支持的算子会 fallback 到 CPU。即使是纯 BPU 模型，输入/输出的量化/反量化节点也由 CPU 执行。可通过 `hrt_model_exec model_info` 查看算子分配情况。
</details>

<details>
<summary><b>5. 如何确认我的板卡使用的 BPU 平台？</b></summary>
<br>

```bash
cat /sys/class/boardinfo/soc_name
```

- `s100` → RDK S100，BPU 为 Nash-e（80 TOPS @ int8）
- `s100p` → RDK S100P，BPU 为 Nash-m（128 TOPS @ int8）
- `s600` → RDK S600，BPU 为 Nash-p
</details>

---

## 社区与贡献

### Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=D-Robotics/rdk_model_zoo&type=Date)](https://star-history.com/#D-Robotics/rdk_model_zoo&Date)

欢迎贡献！请在 [GitHub Issues](https://github.com/D-Robotics/rdk_model_zoo/issues) 提交问题，或在[开发者社区](https://developer.d-robotics.cc/)参与讨论。

## 许可证

本项目遵循 [Apache License 2.0](./LICENSE) 协议。
