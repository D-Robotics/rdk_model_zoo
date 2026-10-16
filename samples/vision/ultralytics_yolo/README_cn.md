[English](README.md) | 简体中文

# Ultralytics YOLO

<a id="overview"></a>

## 概览

本 Sample 为 RDK X5、S100、S100P、S600 提供目标检测、实例分割、姿态估计、分类及 YOLO26 旋转框任务。YOLO 检测头在多个尺度预测类别与框，CPU 端解码、筛选和还原原图坐标；分割、姿态、OBB 还输出各自的掩码、关键点和角度。模型源项目为 [Ultralytics](https://github.com/ultralytics/ultralytics)。

Python 按目标绑定输入、按模型系列选择任务协议，下载、绘图和评估共用。独立 YOLOv5、YOLOE 和 yolo26_depth 使用各自的模型与运行契约。

<a id="directory"></a>
## 目录结构

```text
ultralytics_yolo/
├── conversion/  # 导出与量化配置
├── evaluator/  # 评估程序与指标
├── model/  # 模型文件与下载脚本
├── runtime/  # 推理程序
├── test_data/  # 示例输入
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="support-matrix"></a>
## 支持矩阵

下表列出本 Sample 已发布的模型。单元格为可选模型尺寸，`—` 表示该组合未发布。精度与性能见[评估指南](evaluator/README_cn.md)。

| 模型系列 | 任务 | X5 | S100 | S100P | S600 |
|---|---|---|---|---|---|
| YOLOv5u | detect | n/s/m/l/x | n/s/m/l/x | n/s/m/l/x | n/s/m/l/x |
| YOLOv8 / YOLO11 | detect / seg / pose / cls | n/s/m/l/x | n/s/m/l/x | n/s/m/l/x | n/s/m/l/x |
| YOLOv9 | detect | t/s/m/c/e | t/s/m/c/e | t/s/m/c/e | s/m/c/e |
| YOLOv9 | seg | c/e | c/e | c/e | — |
| YOLOv10 | detect | n/s/m/b/l/x | n/s/m/b/l/x | n/s/m/b/l/x | n/s/m/b/l/x |
| YOLO12 | detect | n/s/m/l/x | n/s/m/l/x | n/s/m/l/x | n/s/m/l/x |
| YOLOv13 | detect | n/s/l/x | — | — | — |
| YOLO26 | detect / seg / pose / cls / obb | n/s/m/l/x | n/s/m/l/x | n/s/m/l/x | n/s/m/l/x |

从[模型清单](model/README_cn.md)选择已发布的目标板卡、模型系列、任务和尺度组合。

运行方式：

- Python 提供上表全部系列/任务组合的入口，见[快速开始](#quickstart)。
- C++ 仅实现其中一部分，范围与输出要求见 [C++ 指南](runtime/cpp/README_cn.md)。已知限制：X5 已发布的分割、姿态、旋转框 `.bin` 制品无法通过 C++ 运行时的输出绑定，需用 Python 入口运行；S 系列 YOLOv10 为无 NMS 接口而 C++ detect 路径固定调用 NMS，S 系列 YOLOv10 也需使用 Python 入口（X5 的 YOLOv10 采用 DFL+NMS）。

| 运行契约 | X5 | S100 / S100P / S600 |
|---|---|---|
| 制品 | `.bin` / bayese | `.hbm` / nashe、nashm、nashp |
| NV12 输入 | 一个 packed 缓冲区 | NHWC Y + UV |
| Python 检测 NMS 默认 | 0.70 | 0.45；YOLOv10 无 NMS |
| 分类 CLI 缩放 | YOLO26 拉伸；其余 letterbox | 拉伸 |

<a id="prerequisites"></a>
## 前置条件

需要完整仓库、Python 3、NumPy、OpenCV、SciPy、PyYAML；实际推理还需要匹配板端系统镜像提供的 `hbm_runtime`。请使用与所选制品配套的板端镜像和 SDK。

推理、下载与主机转换环境分开：C++ 需要板端开发库，ONNX/量化编译需要训练与 OE 环境，详见各子目录。help/list/dry-run/download 可在主机运行。板卡身份来自 [平台注册表](../../../docs/release/platforms.json)。

<a id="quickstart"></a>
## 快速开始

从仓库根目录执行。先联网准备模型，再在匹配的 X5 上运行；输入图片已在仓库 test_data 中。

```bash
bash samples/vision/ultralytics_yolo/model/download_model.sh \
  --platform x5 --family yolov8 --task detect --model-size n
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform x5 --family yolov8 --task detect \
  --model-path samples/vision/ultralytics_yolo/model/yolov8n_detect_bayese_640x640_nv12.bin \
  --label-file datasets/coco/coco_classes.names \
  --test-img samples/vision/ultralytics_yolo/test_data/bus.jpg \
  --img-save-path /tmp/yolov8n-x5.jpg
```
其他目标换用对应的模型准备参数和路径，详见 [模型说明](model/README_cn.md)。显式 `--model-path` 不会自动下载；自定义文件名请明确 `--family`。`--target` 是 `--platform` 的别名；推理会校验板卡身份及目标匹配。

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

检测命令打印所选模型、输入协议及检测信息，并将绘制结果写入 `/tmp/yolov8n-x5.jpg`，成功时打印 `[Saved]`。框采用原图像素坐标；类别 ID 从 0 开始。分类打印 Top-K 而不写图片；分割、姿态、OBB 的结果字段见 [Python 文档](runtime/python/README_cn.md)。检测效果示例：

![Reference detection illustration](test_data/ultralytics_YOLO_Detect_demo.jpg)

S 系列检测效果示例：

![Reference S delivery detection illustration](test_data/result_detect.jpg)

YOLO26 检测效果示例（类别 ID 与分数）：

![Reference S YOLO26 delivery detection illustration](test_data/result_detect_yolo26.jpg)

<a id="entry-points"></a>
## 操作与源码入口

- [model](model/README_cn.md) — 模型下载、清单、路径与摘要.
- [runtime/python](runtime/python/README_cn.md) — 任务 CLI、完整参数表、库接入与前/推/后处理.
- [runtime/cpp](runtime/cpp/README_cn.md) — 按任务构建、位置参数、生命周期与测量范围.
- [conversion](conversion/README_cn.md) — 源权重、导出、标定与目标编译.
- [evaluator](evaluator/README_cn.md) — COCO/ImageNet/DOTA、逐任务评估与参考测量.
- [test_data](test_data/README_cn.md) — 随仓输入图片、显示标签表与效果示例.

在 `runtime/python/` 中，`main.py` 是命令行入口，`cli.py` 负责参数声明、发布资产选择与结果打印，`backend.py` 负责模型加载与张量绑定，任务文件（`detect.py`/`segment.py`/`pose.py`/`obb.py`/`classify.py`）实现各任务的前处理、推理与后处理。检测 DFL 与 YOLO26 direct LTRB 使用不同协议，详见 [检测输入/输出](runtime/python/README_cn.md#detection-io)。输入几何来自模型元数据或显式回退值。YOLO26 OBB 使用弧度；X5 类别内 NMS/裁剪与 S 系列目标不同。

<a id="license"></a>
## 许可与来源

Sample 代码遵循仓库 [Apache-2.0 LICENSE](../../../LICENSE)。模型权重和上游训练框架按其自身许可发布，请分别核对。制品地址和发布方 SHA-256 以清单为准。

<a id="readable-example"></a>
<a id="custom-models"></a>
## 自定义模型

先用[转换流程](conversion/README_cn.md)编译自己训练的模型，再按路径运行：

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform x5 --family yolov8 --task detect \
  --model-path /path/to/my_detect_model.bin \
  --classes-num 80 --label-file /path/to/my_classes.names \
  --test-img samples/vision/ultralytics_yolo/test_data/bus.jpg \
  --img-save-path /tmp/custom-detect.jpg
```

- 显式 `--model-path` 按原样使用，不会触发下载；自定义文件名请同时给出
  `--family`，以便选择匹配的任务协议。
- 未提供 `--label-file` 时结果只显示类别 ID。显式标签文件的条目数与模型
  类别数不符时在推理前报错；自定义 detect/seg/obb 图还需提供匹配的
  `--classes-num`。
- 任务类（`YoloDetect`、`YoloSeg`、`YoloPose`、`YOLO26OBB`、`YoloCls`）可在
  自己的代码中直接调用：`predict` 接受图片路径或 BGR 数组。
  图片路径方式仅检测支持；seg/pose/obb/cls 接受 BGR 数组。
- 数据集指标评估通过[评估器](evaluator/README_cn.md)进行；更多接入示例见
  [docs/architecture/model-examples.md](../../../docs/architecture/model-examples.md)。
