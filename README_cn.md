<div align="center">
  <img src="docs/assets/model_zoo_logo.jpg" width="60%" alt="RDK Model Zoo Logo"/>
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
- 🌐 **全场景覆盖**：涵盖分类、检测、分割、姿态估计、OCR 以及多模态模型。

### 硬件与系统支持

本仓库通过不同分支区分各类板卡的交付内容、历史 demo 和说明文档。当前 `rdk_x5` 分支将作为 RDK X5 的主交付分支；原 `main` 分支已变更为 `rdk_x5_legacy`，仅用于历史 demo 归档。

| 目标硬件 | 对应分支 | 说明 |
| :--- | :--- | :--- |
| RDK X5 | [`rdk_x5`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x5) | RDK X5 主交付分支。推荐系统版本为 RDK OS >= 3.5.0，系统基于 Ubuntu 22.04 aarch64 和 TROS-Humble。 |
| RDK X5 历史 demo | [`rdk_x5_legacy`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x5_legacy) | 原 RDK X5 历史 demo 归档分支，仅用于历史兼容和旧 demo 查询。 |
| RDK X3 | [`rdk_x3`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x3) | RDK X3 设备请切换到该分支。 |
| RDK S 系列 | [`rdk_s`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_s) | RDK S 系列板卡请切换到该分支。RDK S 系列板卡的历史 demo 保留在 [RDK Model Zoo S](https://github.com/d-Robotics/rdk_model_zoo_s) 仓库。 |

---

## 仓库目录结构

<details>
<summary><b>点击展开项目目录结构</b></summary>

<br>

```bash
rdk_model_zoo/
|-- samples/
|   |-- vision/
|   |   |-- clip/                 # 图文多模态匹配
|   |   |-- convnext/             # 图像分类
|   |   |-- edgenext/             # 图像分类
|   |   |-- efficientformer/      # 图像分类
|   |   |-- efficientformerv2/    # 图像分类
|   |   |-- efficient_sam/        # 提示式图像分割
|   |   |-- mobile_sam/           # 提示式图像分割
|   |   |-- efficientnet/         # 图像分类
|   |   |-- efficientvit/         # 图像分类
|   |   |-- fasternet/            # 图像分类
|   |   |-- fastvit/              # 图像分类
|   |   |-- fcos/                 # 目标检测
|   |   |-- googlenet/            # 图像分类
|   |   |-- hgnetv2/              # 图像分类
|   |   |-- lprnet/               # 车牌识别
|   |   |-- mobilenetv1/          # 图像分类
|   |   |-- mobilenetv2/          # 图像分类
|   |   |-- mobilenetv3/          # 图像分类
|   |   |-- mobilenetv4/          # 图像分类
|   |   |-- mobileone/            # 图像分类
|   |   |-- modnet/               # 图像抠图
|   |   |-- paddleocr/            # OCR 文字检测与识别
|   |   |-- repghost/             # 图像分类
|   |   |-- repvgg/               # 图像分类
|   |   |-- repvit/               # 图像分类
|   |   |-- resnet/               # 图像分类
|   |   |-- resnext/              # 图像分类
|   |   |-- unet/                 # 语义分割
|   |   |-- ultralytics_yolo/     # 检测、分割、姿态、分类
|   |   |-- ultralytics_yolo26/   # 检测、分割、姿态、分类
|   |   |-- vargconvnet/          # 图像分类
|   |   |-- yolo26_depth/         # 单目深度估计
|   |   |-- yoloe/                # 实例分割
|   |   |-- yolov5/               # 目标检测
|   |   `-- yoloworld/           # 开放词表目标检测
|-- docs/                  # 项目规范与参考文档
|-- datasets/              # 数据集与下载脚本
|-- tros/                  # TROS 集成指南与示例
|-- utils/                 # 公共 C++ / Python 工具
```
</details>

---

## 快速开始 (Quick Start)

1. **检查系统版本**
   - 确保目标板卡系统版本满足 `RDK OS >= 3.5.0`。
2. **连接硬件**
   - 确保 RDK 板卡上电并可通过 SSH 或 VSCode Remote SSH 访问。
3. **先阅读对应 README**
   - 进入目标目录后先阅读 `README.md` / `README_cn.md`，再执行命令。
4. **运行 Ultralytics YOLO11x 检测 sample**

```bash
cd samples/vision/ultralytics_yolo/model
wget -nc https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/ultralytics_YOLO/yolo11x_detect_bayese_640x640_nv12.bin

cd ../runtime/python
python3 main.py \
  --task detect \
  --model-path ../../model/yolo11x_detect_bayese_640x640_nv12.bin \
  --test-img ../../../../../datasets/coco/assets/bus.jpg \
  --img-save-path ../../test_data/inference_yolo11x.jpg
```

**推理结果示例：**
<div align="center">
  <img src="samples/vision/ultralytics_yolo/test_data/inference_yolo11x.jpg" width="80%" alt="YOLO11x Inference Result"/>
</div>

---

## 模型列表

| 类别 | 模型名称 | 模型路径 | 支持平台 | 详情 |
| :--- | :--- | :--- | :--- | :---: |
| 图像分类 | ConvNeXt | `samples/vision/convnext` | RDK X5 | [详情](./samples/vision/convnext) |
| 图像分类 | EdgeNeXt | `samples/vision/edgenext` | RDK X5 | [详情](./samples/vision/edgenext) |
| 图像分类 | EfficientFormer | `samples/vision/efficientformer` | RDK X5 | [详情](./samples/vision/efficientformer) |
| 图像分类 | EfficientFormerV2 | `samples/vision/efficientformerv2` | RDK X5 | [详情](./samples/vision/efficientformerv2) |
| 图像分类 | EfficientNet | `samples/vision/efficientnet` | RDK X5 | [详情](./samples/vision/efficientnet) |
| 图像分类 | EfficientViT | `samples/vision/efficientvit` | RDK X5 | [详情](./samples/vision/efficientvit) |
| 图像分类 | FasterNet | `samples/vision/fasternet` | RDK X5 | [详情](./samples/vision/fasternet) |
| 图像分类 | FastViT | `samples/vision/fastvit` | RDK X5 | [详情](./samples/vision/fastvit) |
| 图像分类 | GoogLeNet | `samples/vision/googlenet` | RDK X5 | [详情](./samples/vision/googlenet) |
| 图像分类 | HGNetV2 | `samples/vision/hgnetv2` | RDK X5 | [详情](./samples/vision/hgnetv2) |
| 图像分类 | MobileNetV1 | `samples/vision/mobilenetv1` | RDK X5 | [详情](./samples/vision/mobilenetv1) |
| 图像分类 | MobileNetV2 | `samples/vision/mobilenetv2` | RDK X5 | [详情](./samples/vision/mobilenetv2) |
| 图像分类 | MobileNetV3 | `samples/vision/mobilenetv3` | RDK X5 | [详情](./samples/vision/mobilenetv3) |
| 图像分类 | MobileNetV4 | `samples/vision/mobilenetv4` | RDK X5 | [详情](./samples/vision/mobilenetv4) |
| 图像分类 | MobileOne | `samples/vision/mobileone` | RDK X5 | [详情](./samples/vision/mobileone) |
| 图像分类 | RepGhost | `samples/vision/repghost` | RDK X5 | [详情](./samples/vision/repghost) |
| 图像分类 | RepVGG | `samples/vision/repvgg` | RDK X5 | [详情](./samples/vision/repvgg) |
| 图像分类 | RepViT | `samples/vision/repvit` | RDK X5 | [详情](./samples/vision/repvit) |
| 图像分类 | ResNet | `samples/vision/resnet` | RDK X5 | [详情](./samples/vision/resnet) |
| 图像分类 | ResNeXt | `samples/vision/resnext` | RDK X5 | [详情](./samples/vision/resnext) |
| 图像分类 | VargConvNet | `samples/vision/vargconvnet` | RDK X5 | [详情](./samples/vision/vargconvnet) |
| 提示式图像分割 | EfficientSAM-Tiny | `samples/vision/efficient_sam` | RDK X5 | [详情](./samples/vision/efficient_sam) |
| 提示式图像分割 | MobileSAM | `samples/vision/mobile_sam` | RDK X5 | [详情](./samples/vision/mobile_sam) |
| 语义分割 | UNet ResNet 系列 | `samples/vision/unet` | RDK X5 | [详情](./samples/vision/unet) |
| 目标检测 | FCOS | `samples/vision/fcos` | RDK X5 | [详情](./samples/vision/fcos) |
| 目标检测 | YOLOv5 | `samples/vision/yolov5` | RDK X5 | [详情](./samples/vision/yolov5) |
| 目标检测 / 实例分割 / 姿态估计 / 图像分类 | Ultralytics YOLO（`YOLOv5u / YOLOv8 / YOLOv9 / YOLOv10 / YOLO11 / YOLO12 / YOLO13`） | `samples/vision/ultralytics_yolo` | RDK X5 | [详情](./samples/vision/ultralytics_yolo) |
| 目标检测 / 实例分割 / 姿态估计 / 图像分类 | Ultralytics YOLO26 | `samples/vision/ultralytics_yolo26` | RDK X5 | [详情](./samples/vision/ultralytics_yolo26) |
| 单目深度估计 | YOLO26 Depth | `samples/vision/yolo26_depth` | RDK X5 | [详情](./samples/vision/yolo26_depth) |
| 实例分割 | YOLOE | `samples/vision/yoloe` | RDK X5 | [详情](./samples/vision/yoloe) |
| 图像抠图 | MODNet | `samples/vision/modnet` | RDK X5 | [详情](./samples/vision/modnet) |
| OCR 文字检测与识别 | PaddleOCR | `samples/vision/paddleocr` | RDK X5 | [详情](./samples/vision/paddleocr) |
| 车牌识别 | LPRNet | `samples/vision/lprnet` | RDK X5 | [详情](./samples/vision/lprnet) |
| 图文多模态匹配 | CLIP | `samples/vision/clip` | RDK X5 | [详情](./samples/vision/clip) |
| 开放词表目标检测 | YOLOWorld | `samples/vision/yoloworld` | RDK X5 | [详情](./samples/vision/yoloworld) |

## 文档说明与学习资源

为了帮助你更好地理解和使用 RDK 平台与本仓库代码，建议优先阅读以下文档：

- **模型说明**
  - 每个模型目录下的 `README.md` / `README_cn.md` 都包含整体介绍、运行方法和目录说明。
- **源码参考**
  - 如需了解代码级接口说明，请参考 **[源码文档说明](./docs/source_reference/README.md)**。
- **开发规范**
  - 如需新增或开发 Sample，请先阅读 **[Model Zoo 仓库规范指南](./docs/Model_Zoo_Repository_Guidelines.md)**。
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

<a href="https://www.star-history.com/?type=date&repos=D-Robotics%2Frdk_model_zoo">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=D-Robotics/rdk_model_zoo&type=date&theme=dark&legend=top-left&sealed_token=pcz2nn9lRITzBL-JyNLEYFdMZf7Ra0ft7FtCA_eTVEsXH_7xk2cX9jbYWg1AT0ilwVvO4VgrNH0vXv3LVHeGq58Yi24r1novjfb7VFH3Gc1GCT2jGjg38g" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=D-Robotics/rdk_model_zoo&type=date&legend=top-left&sealed_token=pcz2nn9lRITzBL-JyNLEYFdMZf7Ra0ft7FtCA_eTVEsXH_7xk2cX9jbYWg1AT0ilwVvO4VgrNH0vXv3LVHeGq58Yi24r1novjfb7VFH3Gc1GCT2jGjg38g" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=D-Robotics/rdk_model_zoo&type=date&legend=top-left&sealed_token=pcz2nn9lRITzBL-JyNLEYFdMZf7Ra0ft7FtCA_eTVEsXH_7xk2cX9jbYWg1AT0ilwVvO4VgrNH0vXv3LVHeGq58Yi24r1novjfb7VFH3Gc1GCT2jGjg38g" />
 </picture>
</a>

欢迎参与共建 RDK Model Zoo。如有问题或建议，请通过 [GitHub Issues](https://github.com/D-Robotics/rdk_model_zoo/issues) 提出，或在 [D-Robotics 开发者社区](https://developer.d-robotics.cc/) 交流。

## 许可证 (License)

本项目采用 [Apache License 2.0](./LICENSE) 开源协议。
