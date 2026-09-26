# Ultralytics YOLO — RDK X5 / S

[English](README.md)

<a id="overview"></a>
## 概览

本 Sample 为 RDK X5、S100、S100P、S600 提供目标检测、实例分割、姿态估计、分类及 YOLO26 旋转框任务。YOLO 检测头在多个尺度预测类别与框，CPU 端解码、筛选和还原原图坐标；分割、姿态、OBB 还输出各自的 mask、关键点和角度。模型源项目为 [Ultralytics](https://github.com/ultralytics/ultralytics)。

维护入口是 `samples/vision/ultralytics_yolo`。Python 按目标绑定输入、按模型系列选择任务协议，下载、绘图和评估共用；YOLO26 已作为同一 Sample 的一个系列合并。独立 YOLOv5、YOLOE 和 yolo26_depth 具有不同能力，不在此入口中伪装成同一种模型。

<a id="support-matrix"></a>
## 支持范围与验证状态

以下状态按任务/语言区分。supported-verified 只表示已有所列固定输入迁移对照证据，不表示全部精度、性能或当前每个配置均已验证。

| Scope | x5 | s100 | s100p | s600 |
|---|---|---|---|---|
| Python YOLOv8n / YOLO26n detect | supported-verified | supported-verified | supported-verified | supported-verified |
| Python other published scales/tasks | supported-not-run | supported-not-run | supported-not-run | supported-not-run |
| Python YOLOv9 seg c/e | supported-not-run | supported-not-run | supported-not-run | not-supported |
| Python YOLOv13 detect n/s/l/x | supported-not-run | not-supported | not-supported | not-supported |
| C++ detect/classify/pose/segment reference contracts | supported-not-run | supported-not-run | supported-not-run | supported-not-run |
| C++ OBB | not-supported | not-supported | not-supported | not-supported |

Python 发布组合详见 [模型清单](model/README_cn.md)：YOLOv5u/v10/12 为检测；YOLOv8/11 为 detect/seg/pose/cls；YOLOv9 分割仅 c/e，S600 无 t 检测及分割；YOLOv13 仅 X5。YOLO26 全目标各 25 个资产（五任务 × n/s/m/l/x），是既有资产合并，不是新增 100 个模型。C++ 范围以其 [输入/head 契约及限制](runtime/cpp/README_cn.md) 为准，不能直接套用 Python 全清单。

历史检测证据：[P1](../../../docs/releases/unified-migration/2026-09-16-pilot-validation.md)、[P2](../../../docs/releases/unified-migration/2026-09-16-p2-validation.md)，覆盖 X5 8GB/4GB 与 S 三目标的 YOLOv8n、YOLO26n。其他任务/尺度、数据集精度、性能和本地转换不据此扩大。原始指标保留在 [X5 评估文档](../../../platforms/x5/samples/vision/ultralytics_yolo/evaluator/README.md) 与 [S 评估文档](../../../platforms/s/samples/vision/ultralytics_yolo/evaluator/README.md)，不是重构后的重测结果。

| 运行契约 | X5 | S100 / S100P / S600 |
|---|---|---|
| 制品 | `.bin` / bayese | `.hbm` / nashe、nashm、nashp |
| NV12 输入 | 一个 packed 缓冲区 | NHWC Y + UV |
| Python 检测 NMS 默认 | 0.70 | 0.45；YOLOv10 无 NMS |
| 分类 CLI resize | YOLO26 拉伸；其余 letterbox | 拉伸 |
| 分类文件名标记（不是输入覆盖） | YOLO26 224，其余 640 | 公开 URL 使用 224；S100/S100P v8/v11 保留 640 兼容 ID |

<a id="prerequisites"></a>
## 前置条件

需要完整仓库、Python 3、NumPy、OpenCV、SciPy、PyYAML；实际推理还需要匹配板端系统镜像提供的 `hbm_runtime`。不会静默安装依赖。具体镜像/SDK 版本以所用制品和上述验证证据为准，本 Sample 未为全部目标给出一个统一的最低镜像版本。选择大尺度模型前确认存储和内存容量；本轮未测全尺度峰值内存，不能以下载成功推断运行可行。

推理、下载与主机转换环境分开：C++ 需要板端开发库，ONNX/量化编译需要训练与 OE 环境，详见各子目录。主机可使用 help/list/dry-run/download；它们不执行板端推理。硬件身份来自 [统一注册表](../../../docs/release/platforms.json)，身份可识别不等于制品已发布或已验证。

<a id="quickstart"></a>
## 快速开始

从仓库根目录执行。先联网准备模型，再在匹配的 X5 上运行；输入图片已在仓库 test_data 中。

```bash
bash samples/vision/ultralytics_yolo/model/download_model.sh \
  --platform x5 --family yolov8 --task detect --model-size n
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform x5 --family yolov8 --task detect \
  --model-path samples/vision/ultralytics_yolo/model/yolov8n_detect_bayese_640x640_nv12.bin \
  --test-img samples/vision/ultralytics_yolo/test_data/bus.jpg \
  --img-save-path /tmp/yolov8n-x5.jpg
```
其他目标换用对应的模型准备参数和路径，详见 [模型说明](model/README_cn.md)。显式 `--model-path` 不会自动下载；省略时历史兼容入口会下载缺失的默认资产。自定义文件名请明确 `--family`。`--target` 是 `--platform` 的别名；真正推理会拒绝未知板卡及目标不匹配。

无板卡时可检查选择结果，不下载也不推理：

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform s100 --list-models
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform s600 --family yolo26 --task cls --dry-run
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --target x5 --task detect \
  --asset-id x5:ultralytics_yolo:yolov8n_detect_bayese_640x640_nv12.bin --dry-run
```
<a id="expected-results"></a>
## 预期结果

检测命令打印所选模型、输入协议及检测信息，并将绘制结果写入 `/tmp/yolov8n-x5.jpg`，成功时打印 `[Saved]`。框采用原图像素坐标；类别 ID 从 0 开始。分类打印 Top-K 而不写图片；分割、姿态、OBB 的结果字段见 [Python 文档](runtime/python/README_cn.md)。模型绑定失败时不能把旧图片当本次结果。下面是保留的历史检测示意图，不是本轮新测量：

![Historical detection illustration](test_data/ultralytics_YOLO_Detect_demo.jpg)

<a id="directory"></a>
## 目录职责

```text
ultralytics_yolo/
├── model/          # published asset preparation
├── runtime/python/ # shared task APIs and CLI
├── runtime/cpp/    # detect/classify/pose/segment reference programs
├── conversion/    # export, calibration and X5/S compiler adapters
├── evaluator/     # dataset evaluation and batch CLI
├── test_data/     # input images, labels and historical illustrations
└── tests/         # host regression checks
```
<a id="entry-points"></a>
## 操作与源码入口

- [model](model/README_cn.md) — 模型下载、清单、路径与摘要.
- [runtime/python](runtime/python/README_cn.md) — 任务 CLI、完整参数表、库接入与前/推/后处理.
- [runtime/cpp](runtime/cpp/README_cn.md) — 按任务构建、位置参数、生命周期与测量范围.
- [conversion](conversion/README_cn.md) — 源权重、导出、标定、编译与未验证前提.
- [evaluator](evaluator/README_cn.md) — COCO/ImageNet/DOTA、逐任务评估与原始指标.

源码阅读顺序：`main.py` 处理参数/文件/显示，`yolo_dispatch.py` 选任务，runner/binding 负责 SDK 与张量，任务类处理前处理、推理、后处理。检测 DFL 与 YOLO26 direct LTRB 不可互换；详见 [检测契约](DETECTION_CONTRACT.md)。输入尺寸必须由元数据/显式回退正确解析，不能按文件名猜测。YOLO26 OBB 使用弧度，X5 类别内 NMS/裁剪与 S 路径有区别；修正后的非检测行为仍需板端精度复验。

旧 `platforms/{x5,s}/samples/vision/ultralytics_yolo` 与 `ultralytics_yolo26` 入口保留转发兼容；历史表格、URL 和来源记录保留。不要把其他尚未迁移 Sample 的状态推断为本入口已支持。新增任务必须先明确输出契约、提供主机回归，再更新双语 README 和验证范围。

<a id="license"></a>
## 许可与来源

Sample 代码遵循仓库 [Apache-2.0 LICENSE](../../../LICENSE)，保留各文件原版权声明。模型权重和上游训练框架按其随附许可分别核对；仓库代码许可不自动授予所有权重同样的许可。制品地址和发布方摘要以清单为准，缺少发布方哈希时不能把本地摘要当作来源认证。
