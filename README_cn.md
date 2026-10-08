![RDK Model Zoo](docs/assets/model_zoo_logo.jpg)

[English](README.md) | 简体中文

# ⭐️ 点个Star不迷路, 感谢您的关注 ⭐️

RDK Model Zoo 提供在地瓜机器人 RDK 板卡上运行的视觉、语音和大模型示例，包含模型下载、转换、推理与评估程序。源码版本为 2.0.0，见根目录 `VERSION`。

## 目录结构

```text
samples/                  # 任务实现与指南
docs/release/             # 制品/基准事实与目标身份
docs/sample-standards/    # README 与推理契约
docs/architecture/        # 可读 Runtime 架构
docs/validation/          # 板端冒烟测试清单
datasets/                 # 数据集入口
utils/                    # Python/C++ 公共函数
tools/                    # 目录、契约检查与验证工具
```

## 按任务开始

完整清单在 [Sample 索引](samples/README_cn.md)，目前包含 51 个 Sample：45 个视觉、3 个语音、1 个机器人策略和 2 个大模型样例。每个入口说明自己的 target、变体和语言。

| 任务 | 入口 |
|---|---|
| 检测、分割、姿态、分类、旋转框 | [Ultralytics YOLO](samples/vision/ultralytics_yolo/README_cn.md) |
| 无提示实例分割 | [YOLOE](samples/vision/yoloe/README_cn.md) |
| 语音识别 | [ASR](samples/speech/asr/README_cn.md)、[Paraformer](samples/speech/paraformer/README_cn.md) |
| 唤醒词检测 | [KWS](samples/speech/kws/README_cn.md) |
| 离线机器人策略 | [HIMLoco](samples/robotics/himloco/README_cn.md)：六帧观测到动作，不执行实机控制 |
| 视觉语言模型 | [Gemma4-E2B](samples/llm/gemma4-e2b/README_cn.md)：原生对话、HTTP、单次推理及验证工具 |
| 文本生成 | [MiniCPM5-2B](samples/llm/minicpm5-2b/README_cn.md)：S100/S100P OELLM 1.0.0 与 S600 OELLM 2.0 beta 原生入口 |
| 图像分类 | [ResNet](samples/vision/resnet/README_cn.md)、MobileNet、EfficientNet、ConvNeXt、Rep 系列等，见完整索引 |
| 文本检测与识别 | [PaddleOCR](samples/vision/paddle_ocr/README_cn.md) |
| 提示分割 | [EfficientSAM](samples/vision/efficient_sam/README_cn.md)、[MobileSAM](samples/vision/mobile_sam/README_cn.md) |
| 检测、开放词汇与跟踪 | [YOLOv5](samples/vision/yolov5/README_cn.md)、[FCOS](samples/vision/fcos/README_cn.md)、[YOLOWorld](samples/vision/yoloworld/README_cn.md)、[ByteTrack](samples/vision/bytetrack/README_cn.md) |
| 车牌识别、抠图 | [LPRNet](samples/vision/lprnet/README_cn.md)、[MODNet](samples/vision/modnet/README_cn.md) |
| 点云部件分割 | [PointNet](samples/vision/pointnet/README_cn.md) |
| 语义分割 | [UNet](samples/vision/unet/README_cn.md) · [PP-LiteSeg](samples/vision/pp_liteseg/README_cn.md) · [UNetMobileNet](samples/vision/unetmobilenet/README_cn.md) |
| 单目深度 | [YOLO26 Depth](samples/vision/yolo26_depth/README_cn.md) · [Depth Anything V2](samples/vision/depth_anything_v2/README_cn.md) |
| 车道嵌入与二值标签 | [LaneNet](samples/vision/lanenet/README_cn.md) |
| 准备特征的轨迹规划 | [DiffusionDrive](samples/vision/diffusiondrive/README_cn.md) |
| 视觉特征、图文匹配、视频分类 | [DINOv2](samples/vision/dinov2/README_cn.md)、[SigLIP](samples/vision/siglip/README_cn.md)、[CLIP](samples/vision/clip/README_cn.md)、[3DResNet](samples/vision/3dresnet/README_cn.md) |

ACT／Pi0 以固定上游 Git 子模块集成，用于视觉-语言-动作策略：入口见 [VLA 指南](samples/vla/README_cn.md)。ACT S100 与 S600 使用不同源版本，Pi0 面向 S600，模型资源由使用者准备。

## 板卡、制品与环境

| 目标 | 制品 | March | 说明 |
|---|---|---|---|
| RDK X5 | `.bin` | bayes-e | 有 4GB/8GB 内存版本 |
| RDK S100 | `.hbm` | nash-e | 不能替代 S100P/S600 制品 |
| RDK S100P | `.hbm` | nash-m | 只有明确发布的组合才可运行，无资产不回退 S100 |
| RDK S600 | `.hbm` | nash-p | 输入形状/模型组合按实际制品解析 |

各 Sample 通过自己的原生 CLI 选择板卡——ResNet 使用 `--target auto|x5|s100|s100p|s600`，YOLO 使用 `--platform`；`auto` 依据系统板卡身份（SoC 名称/板型，辅以 socinfo 与设备树）解析目标，无法识别的板卡显式报错。使用匹配板卡镜像提供的 SDK（`hbm_runtime` 模块随该镜像提供）。主机和板端依赖、镜像版本及模型内存要求以具体 Sample 为准。Python 常见依赖为 NumPy、OpenCV、SciPy、PyYAML；并非每个任务都使用相同输入协议或安装集。C++ 需要板端开发头/库，模型转换需要主机训练/OE 环境。

## 快速开始

克隆完整仓库，先在相应 Python 环境中准备 Sample 文档列明的依赖。普通推理无需初始化 VLA 子模块。

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
  --label-file datasets/coco/coco_classes.names \
  --test-img samples/vision/ultralytics_yolo/test_data/bus.jpg \
  --img-save-path /tmp/rdk-yolov8n.jpg
```
成功时打印 `[Saved]` 并生成 `/tmp/rdk-yolov8n.jpg`。图片来自仓库 `test_data`。其他板卡须选择对应 target 与制品路径；没有板卡时可用 `--dry-run` 检查参数解析。模型来源、离线复制、哈希限制及更多任务见 [YOLO 模型说明](samples/vision/ultralytics_yolo/model/README_cn.md) 和 [运行说明](samples/vision/ultralytics_yolo/runtime/python/README_cn.md)。首次分类运行见 [ResNet 快速开始](samples/vision/resnet/README_cn.md#quickstart)；板端测试检查清单见[板端冒烟测试](docs/validation/board-smoke-test.md)。

## 阅读和扩展代码


从 Sample 的 `runtime/python/main.py` 开始阅读：入口构造模型，再调用 `predict`。预处理、推理和后处理位于模型文件中；参数解析、模型选择和结果展示位于 `cli.py` 或 `yolo_cli.py`。

例如，ResNet 的 [classify.py](samples/vision/resnet/runtime/python/classify.py) 实现图像分类，YOLO 的 [detect.py](samples/vision/ultralytics_yolo/runtime/python/detect.py) 实现目标检测。多模型流水线和跟踪、解码等算法使用各自的模块。模型加载和 SDK 调用复用 [Python 公共函数](utils/py_utils/README_cn.md)。两个 `samples/llm` 示例使用原生 C++ 接口。

自定义模型的导出与编译见各 Sample 的 `conversion/`，数据集评估见 `evaluator/`。新增 Sample 时参照 [Runtime 代码规范](docs/sample-standards/runtime-code.md)、[推理契约](docs/sample-standards/inference-contract.md) 和 [README 规范](docs/sample-standards/readme-contract.md)，贡献代码前阅读 [AGENTS.md](AGENTS.md)。

## 数据与验证

[模型性能数据](docs/benchmarks/README_cn.md)按任务和板卡列出分类精度、BPU 吞吐量与后处理时间。

- [模型发布清单](docs/release) 包含模型文件、下载地址和性能数据。X5 与 S 系列清单位于 `docs/release/{x5,s}/models.yaml`，板卡标识位于 `docs/release/platforms.json`。
- 数据准备入口：[datasets](datasets)。大数据集和模型通常不在 Git 中。
- [TROS](docs/tros/README_cn.md) 板端运行栈文档。
- Sample 指南中的性能表记录已发布测量值及其测试条件；数据集精度评估见各 Sample 的 `evaluator/` 指南。

## 模型目录数据（维护者）

`tools/catalog-publisher` 校验各平台清单并生成派生数据包。使用其 package.json 要求的 Node 22.12+、低于 23。从仓库根目录执行：

```bash
npm --prefix tools/catalog-publisher ci
npm --prefix tools/catalog-publisher run check
npm --prefix tools/catalog-publisher run catalog:build
```

## 社区、贡献与许可

- [在线模型目录](https://d-robotics.github.io/rdk_model_zoo/)
- [GitHub Issues](https://github.com/D-Robotics/rdk_model_zoo/issues)：请附 target、系统/SDK、模型引用、提交及可复现命令；先保护原始失败输出，再提交代码与相应文档/测试。
- [D-Robotics 开发者社区](https://developer.d-robotics.cc/) 与 [RDK 用户手册](https://developer.d-robotics.cc/information)

统一代码见根 [LICENSE](LICENSE)。模型权重、数据集和上游项目按各自许可及来源记录处理。
