# Model Zoo
## 仓库简介

本仓库是 D-Robotics（地瓜机器人）官方提供的 BPU 模型示例与工具集合（Model Zoo），面向 运行在 BPU（Brain Processing Unit）上的 AI 模型部署与应用开发，用于帮助开发者 快速上手 BPU、快速跑通模型推理流程。

仓库中收录了覆盖多个 AI 领域（如计算机视觉、语音）的 BPU 可运行模型，并提供从 模型准备 → 推理运行 → 结果解析 → 示例验证 的完整参考实现，帮助用户以最小成本理解并使用 BPU 能力。

当前主要支持的平台包括：

- RDKS100

- RDKS600

仓库核心价值

- 🚀 快速把 BPU 用起来

    提供可直接运行的模型与示例工程，帮助用户在最短时间内完成 BPU 推理验证。

- 🧩 完整端到端示例

    覆盖模型加载、前处理、BPU 推理执行、后处理与结果可视化，支持 C/C++ 与 Python 双语言接口。

- 📐 规范化设计与完整接口文档

    提供统一的目录结构与示例代码规范，并配套详细的接口文档与使用说明，便于客户快速理解、二次开发，降低集成与维护成本。


## 仓库目录结构

本仓库采用分层清晰、职责明确的目录结构，以便用户快速定位所需内容并开始使用。其中，samples/ 是仓库的核心，集中提供运行在 BPU 上的各类模型示例；docs/ 提供了本仓库的规范说明及接口层面的说明文档；datasets/ 用于存放示例与评测所需的数据集；utils/ 则沉淀了可复用的通用工具，便于批量维护。

以下展示顶层目录结构，用于快速理解仓库整体组织方式：
```bash
.
|-- datasets                               # 公共数据集与示例数据
|-- docs                                   # 项目文档与用户指南
|-- samples                                # 模型示例（核心内容）
|-- tools                                  # 转换/构建/辅助工具
|-- tros                                   # TROS/ROS 相关适配
|-- utils                                  # 通用工具库
|-- LICENSE                                # 许可证
`-- README.md                              # 顶层说明

```

## 快速开始

本仓库中的模型均已按领域进行分类，并汇总在下方的 模型列表 中。
用户可通过如下步骤快速运行目标模型：
- 根据自身需求，在模型列表中查找目标模型；
- 根据表格中提供的路径进入对应的模型目录；
- 进入模型目录后，阅读该目录下的 README.md，其中包含该模型的功能说明、使用方式以及完整的运行指引；


以YOLOv5为例
- 在下方模型列表中定位 YOLOv5；
- 进入模型目录：

    ```bash
    cd samples/vision/yolov5
    ```
- 阅读相应的README文档

- 即可按照模型自身的说明完成推理示例的运行与验证。


如需对本仓库的整体结构、BPU 使用方式及接口能力进行系统了解，建议参考：
```bash
docs/Model_Zoo_User_Guide.md
```

## 模型列表

下表按 应用领域 对当前仓库中已提供的模型进行分类，方便快速查找与定位。
每个模型的详细说明、使用方法和示例，请点击对应的 详情链接 查看该模型目录下的 README.md。

| 类别         | 模型名称 | 模型路径                     | 详情 |
|--------------|----------|------------------------------|------|
| 目标检测     | YOLOv5x  | samples/vision/yolov5        | [README](samples/vision/yolov5/README.md) |
| 图像分类     | ResNet50 | samples/vision/resnet50      | [README](samples/vision/resnet50/README.md) |
| 语音识别     | ASR-XXX  | samples/speech/asr_xxx       | [README](samples/speech/asr_xxx/README.md) |


## 文档说明

- 每个模型的顶层目录的`README.md`中，都有该模型的整体介绍，如果想快速了解某个模型，可直接到相关目录查看；
- 每个模型都有详细的接口介绍，如关系代码层面的接口信息，可阅读[源码文档说明](docs/source_reference/README.md)，根据介绍构建或浏览代码文档；
- 如需提交自己的模型Sample，请于仔细阅读[rdk_model_zoo](docs/Model_Zoo_Repository_Guidelines.md)仓库规范；


## 许可证
[Apache License 2.0](LICENSE)
