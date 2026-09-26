# RDK Model Zoo

[English](README.md)

RDK Model Zoo 为地瓜机器人 BPU 提供模型准备、前处理、推理、后处理及应用验证示例。每个 Sample 的 README 是用户与 Agent 共同的操作入口，包含运行命令、输入输出、源码结构、转换/评估流程及已知限制。运行推理不需要 Agent、Node.js 或模型目录发布工具。

> 当前 `develop` 是 X5/S 全量整合中的开发分支，尚不是已完成迁移的客户发布版。客户交付仍应选择匹配的已发布平台分支/标签及其文档。整合树已超出最初三个试点；源码迁移、主机验证、板测、转换和发布就绪分别记录，不能相互代替。

## 按任务开始

完整清单在 [Sample 索引](samples/README_cn.md)，目前包含 42 个统一视觉 Sample。每个入口说明自己的 target、变体、语言和验证范围。

| 任务 | 统一入口 |
|---|---|
| 检测、分割、姿态、分类、旋转框 | [Ultralytics YOLO](samples/vision/ultralytics_yolo/README_cn.md) |
| 图像分类 | [ResNet](samples/vision/resnet/README_cn.md)、MobileNet、EfficientNet、ConvNeXt、Rep 系列等，见完整索引 |
| 文本检测与识别 | [PaddleOCR](samples/vision/paddle_ocr/README_cn.md) |
| 提示分割 | [EfficientSAM](samples/vision/efficient_sam/README_cn.md)、[MobileSAM](samples/vision/mobile_sam/README_cn.md) |
| 检测、开放词汇与跟踪 | [YOLOv5](samples/vision/yolov5/README_cn.md)、[FCOS](samples/vision/fcos/README_cn.md)、[YOLOWorld](samples/vision/yoloworld/README_cn.md)、[ByteTrack](samples/vision/bytetrack/README_cn.md) |
| 车牌识别、抠图 | [LPRNet](samples/vision/lprnet/README_cn.md)、[MODNet](samples/vision/modnet/README_cn.md) |
| 点云部件分割 | [PointNet](samples/vision/pointnet/README_cn.md) |
| 语义分割 | [UNet](samples/vision/unet/README_cn.md) · [PP-LiteSeg](samples/vision/pp_liteseg/README_cn.md) · [UNetMobileNet](samples/vision/unetmobilenet/README_cn.md) |
| 单目深度 | [YOLO26 Depth](samples/vision/yolo26_depth/README_cn.md) · [Depth Anything V2](samples/vision/depth_anything_v2/README_cn.md) |
| 视觉特征、图文匹配、视频分类 | [DINOv2](samples/vision/dinov2/README_cn.md)、[SigLIP](samples/vision/siglip/README_cn.md)、[CLIP](samples/vision/clip/README_cn.md)、[3DResNet](samples/vision/3dresnet/README_cn.md) |

深度、语义分割、点云、语音、机器人、大模型/VLA 及其他尚未统一的能力仍可从 [X5 原平台入口](platforms/x5/README_cn.md)、[S 原平台入口](platforms/s/README_cn.md) 查阅。待迁移不等于源能力被删除；后续批次见 [迁移台账](docs/releases/unified-migration/x5-s-migration-map.md)。

## 板卡、制品与环境

| 目标 | 制品 | 区别 |
|---|---|---|
| RDK X5 | `.bin`，bayes-e | X5 4GB/8GB 的内存容量不同；验证需区分板位 |
| RDK S100 | `.hbm`，nash-e | 不能替代 S100P/S600 制品 |
| RDK S100P | `.hbm`，nash-m | 只有明确发布的组合才可运行，无资产不回退 S100 |
| RDK S600 | `.hbm`，nash-p | 输入形状/模型组合按实际制品解析 |
| RDK X3 | 历史目录 | 保留原始文档与发布记录，不属于本轮新增适配目标 |

使用匹配板卡镜像提供的 SDK；不能把同名 `hbm_runtime` 当作跨平台通用安装包。主机和板端依赖、镜像版本及模型内存要求以具体 Sample 为准。Python 常见依赖为 NumPy、OpenCV、SciPy、PyYAML；并非每个任务都使用相同输入协议或安装集。C++ 需要板端开发头/库，模型转换需要主机训练/OE 环境。

## 快速开始：查看整合树并运行一个示例

以下是贡献者/评审查看 `develop` 的路径，不代替客户发布选择。保留完整仓库；先在相应 Python 环境中准备 Sample 文档列明的依赖。无需为普通推理初始化 VLA 子模块。

```bash
git clone --branch develop https://github.com/D-Robotics/rdk_model_zoo.git
cd rdk_model_zoo
python3 samples/vision/ultralytics_yolo/runtime/python/main.py --help
python3 samples/vision/ultralytics_yolo/runtime/python/main.py --platform x5 --list-models
```
接着在匹配的 X5 上，从仓库根目录显式准备模型并运行检测：

```bash
bash samples/vision/ultralytics_yolo/model/download_model.sh \
  --platform x5 --family yolov8 --task detect --model-size n
python3 samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform x5 --family yolov8 --task detect \
  --model-path samples/vision/ultralytics_yolo/model/yolov8n_detect_bayese_640x640_nv12.bin \
  --test-img samples/vision/ultralytics_yolo/test_data/bus.jpg \
  --img-save-path /tmp/rdk-yolov8n.jpg
```
成功时打印 `[Saved]` 并生成 `/tmp/rdk-yolov8n.jpg`。图片来自仓库 `test_data`。其他板卡须选择对应 target 与制品路径；没有板卡时使用 `--dry-run`，不能以主机解析通过代替推理。模型来源、离线复制、哈希限制及更多任务见 [YOLO 模型说明](samples/vision/ultralytics_yolo/model/README_cn.md) 和 [运行说明](samples/vision/ultralytics_yolo/runtime/python/README_cn.md)。

## 阅读和扩展代码

```text
samples/                  # unified task implementations and guides
platforms/{x5,s}/         # retained source material and compatibility entries
platforms/x3/             # historical X3 distribution
docs/release/             # artifact/benchmark facts and target identity
docs/sample-standards/    # README and inference contracts
docs/releases/           # migration ledger, reviews and validation evidence
datasets/                # dataset entry points
utils/                   # compatibility utilities
tools/                   # catalog, contract checks and validation tooling
```
`main.py` 负责 CLI、文件和显示；模型任务模块负责前处理、推理、后处理及可选 `predict`；runner/binding 隔离 SDK 与张量契约。`conversion/`、`evaluator/` 各自保存可操作说明。共享机制只放已被多个样例需要的通用能力；OCR 字典、DFL/LTRB 等真实差异保留。目录规范不意味着所有样例已经完成同等程度的审核。

开发前阅读 [AGENTS.md](AGENTS.md)、[推理契约](docs/sample-standards/inference-contract.md)、[README 契约](docs/sample-standards/readme-contract.md)。用户和 Agent 使用同一原生命令，不为 Agent 添加另一套隐藏执行入口。

## 数据、原分支资料与验证

- [统一发布事实](docs/release) 保存制品与历史测量事实；[平台注册表](platforms/README_cn.md) 说明原分支、目录、标签及运行时差异。
- 数据准备入口：[datasets](datasets)、[X5 datasets](platforms/x5/datasets)、[S datasets](platforms/s/datasets)。大数据集和模型通常不在 Git 中。
- 保留原指南：[X5 开发规范](platforms/x5/docs/Model_Zoo_Repository_Guidelines.md)、[S Python API](platforms/s/docs/Python_API_User_Guide.md)、[S UCP](platforms/s/docs/UCP_User_Guide.md)、[TROS](docs/tros/README_cn.md)。
- [迁移台账](docs/releases/unified-migration/x5-s-migration-map.md) 与各批报告说明“实现、主机测试、板测、独立评审、关闭”各状态；[当前非板端计划](docs/superpowers/plans/2026-09-26-host-completion.md) 不把待补板测当作通过。

SAM 已有部分板端证据，不能再笼统写成“全部未测”；同样，单个 YOLO、分类或视频片段通过也不能推广为所有 target/变体、数据集精度或发布就绪。应读取 Sample 及证据中精确的提交、输入、制品和范围。

## 常见问题

**能否在开发电脑上直接运行 BPU 推理？** help、清单、dry-run 和主机测试可在准备好依赖的电脑上运行；BPU 推理需要匹配板端 SDK 和硬件。

**模型能否跨目标复用？** 不能仅靠改后缀、改文件名或换 target。核对 march、输入布局/类型与输出协议；缺少发布资产就明确保留缺口。

**评估表是否代表当前重构代码？** 历史表保留其原测试条件。新代码须有相应的测试记录，主机测试、单图对照、数据集精度和性能测量不能互相代替。

## 模型目录数据（维护者）

`tools/catalog-publisher` 校验各平台清单并生成派生数据包。使用其 package.json 要求的 Node 22.12+、低于 23；这不是板端推理依赖。从仓库根目录执行：

```bash
npm --prefix tools/catalog-publisher ci
npm --prefix tools/catalog-publisher run check
npm --prefix tools/catalog-publisher run catalog:build
```
生成的 `dist/catalog.meta.json` 用 SHA-256 绑定 `catalog.json`；CI 上传数据制品。目录数据、网页发布和板端 Sample 是不同交付，不因代码迁移自动发布新模型或更新历史标签。

## 社区、贡献与许可

问题反馈可使用仓库 Issues，并附 target、系统/SDK、模型引用、提交及可复现命令。先保护原始失败输出，再提交代码与相应文档/测试。原分支保留了[社区资源](platforms/x5/README_cn.md)和平台专属说明。

统一代码见根 [LICENSE](LICENSE)，平台分发另保留 [X5 LICENSE](platforms/x5/LICENSE) 与 [S LICENSE](platforms/s/LICENSE)；X3 上游没有随附许可文件，此处未补造。模型权重、数据集和上游项目按各自许可及来源记录处理。
