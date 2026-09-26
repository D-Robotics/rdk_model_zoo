[English](README.md) | [简体中文](README_cn.md)

# YOLO26 Depth

<a id="overview"></a>
## 概述

从单张 BGR 图像估计稠密相对深度，在同一样例中保留 X5 与 S 的源能力：四种目标的 Python 推理、
X5 C++、显式模型准备、两套转换工具链及离线评估。
绘图、文件读写、模型绑定和 SDK 资源管理与三个推理阶段分离，人和 Agent 使用同样的原生命令。

已发布方案取决于目标：X5 全变体及 S n/s/m 输入 letterbox NV12，输出已校准 log-depth；
S l/x 输入归一化 RGB featuremap，输出原始 logit，只有后者需要 CPU clip/scale/bias。
两者最终都输出原图尺寸的**相对深度**，不是经过校准的米制距离。

<a id="support-matrix"></a>
## 支持矩阵

| 目标 | march | n/s/m | l/x | 运行时 |
|---|---|---|---|---|
| X5 | bayes-e | NV12 / 已校准 log | NV12 / 已校准 log | Python + C++ |
| S100 | nash-e | NV12 / 已校准 log | featuremap / 原始 logit | Python |
| S100P | nash-m | NV12 / 已校准 log | featuremap / 原始 logit | Python |
| S600 | nash-p | NV12 / 已校准 log | featuremap / 原始 logit | Python |

所有目标默认 n，共有 20 个清单制品，包含五个独立 S100P 制品。
源可用性不等于新运行时验收：本轮已做主机测试，板端推理、真实 SDK/OpenCV 原生构建、Torch 导出、
OE 编译和数据集测量仍为 not-run。不声明 S 原生深度实现。
历史源结果见下文与[评估目录](evaluator/README_cn.md)，其中的证据边界明确保留。

<a id="prerequisites"></a>
## 前提

Python 推理需要对应板卡 Linux 镜像及厂商 `hbm_runtime`，另需 Python、NumPy、OpenCV、PyYAML。
主机 list/dry-run 不需要板端 SDK。模型显式准备，推理不下载、不安装软件；加载前核对具体板卡身份。
X5 C++ 依赖与构建见[原生运行时](runtime/cpp/README_cn.md)，转换使用独立准备的 x86 工具链环境。

以下命令从仓库根目录执行。下载需要网络，推理使用本地制品。
摘要和外部路径见[模型准备](model/README_cn.md)。附带 [bus.jpg](test_data/bus.jpg) 只是示例，不是数据集。

<a id="quickstart"></a>
## 快速开始

任意主机上查看制品和选择结果：

```bash
python -m samples.vision.yolo26_depth.runtime.python.main --list-models
python -m samples.vision.yolo26_depth.runtime.python.main --target s100p --variant l --dry-run
```

在实际 X5 板上准备并执行：

```bash
bash samples/vision/yolo26_depth/model/download.sh --target x5 --variant n
bash samples/vision/yolo26_depth/runtime/python/run.sh --target x5 --variant n \
  --test-img samples/vision/yolo26_depth/test_data/bus.jpg --output /work/depth/x5-n
```

S 选择精确目标和变体，例如 S600 lite l：

```bash
bash samples/vision/yolo26_depth/model/download.sh --target s600 --variant l
bash samples/vision/yolo26_depth/runtime/python/run.sh --target s600 --variant l \
  --output /work/depth/s600-l
```

每次使用新输出目录。其他 S 板将 `s600` 换成 `s100` / `s100p`，选择其对应制品。
`--target auto` 使用实际板卡身份，或由精确 `--asset-id` 推断；不会在普通主机上猜测目标。
旧 shell 位置变体参数（`run.sh l`）改为显式 `--variant l`，归档源脚本保留历史接口。

<a id="expected-results"></a>
## 输出与历史参考

Python 写出 `log_depth.npy`（192×192 F32）、`depth_native.npy`（原图 H×W F32）、
`depth.png`、`overlay.png` 和 `report.json`；S lite 另写 `raw_logit.npy`。
X5 原生还保留 `depth_native.f32`。绘图采用 2%/98% 分位范围、反向 TURBO，
叠加权重为原图 0.45 / 深度颜色 0.55。报告包含模型选择和实际本地摘要；SDK 未提供版本时记为 `unknown`。
图像看起来合理不代表精度或米制距离正确。

源 S 根目录给出以下混合方案单图表，作为**未重测的历史信息**保留：

| 变体 | 方案 | raw cosine vs FP32 | S100 延迟 ms | S100P 延迟 ms | S600 延迟 ms |
|---|---|---:|---:|---:|---:|
| n | NV12 | 0.9996 | — | — | — |
| s | NV12 | 0.9984 | — | — | — |
| m | NV12 | 0.9996 | — | — | — |
| l | lite | 0.9997 | 11.0 | 8.1 | — |
| x | lite | 0.9997 | 20.6 | 13.7 | 10.8 |

同页“全部通过 ≥0.999”与 s=0.9984 矛盾。其 evaluator 还有另一组未完整逐行绑定制品的延迟表，
不能拼成一套基准或据此推断验收。X5 HRT 延迟/FPS 及 S 另一张表完整保留在 evaluator README。
源图边界描述也曾把 exp/resize 写到图内，而实际代码在 CPU 执行。
[源审计](../../../docs/releases/unified-migration/2026-09-26-b8-yolo26-depth-source-review.md)记录这些差异和迁移决定。

<a id="directory"></a>
## 目录

```text
yolo26_depth/
├── model/                 # 基于清单的显式下载
├── runtime/python/        # 阶段、逐次上下文、绑定、懒加载 runner、CLI、绘图
├── runtime/cpp/           # X5 阶段 API、SDK owner、张量/IO、启动器
├── conversion/            # 导出、校准、29 份源 YAML、X5/S 编译
├── evaluator/             # 三种输入准备协议、离线指标和比较
├── test_data/bus.jpg      # 逐字节保留的源图像
└── tests/                 # 主机测试与原生纯逻辑/模拟 SDK 检查
```

<a id="entry-points"></a>
## 各入口与集成

- [模型](model/README_cn.md)：全部制品、显式准备及来源。
- [Python](runtime/python/README_cn.md)：CLI、三阶段 API、张量契约和错误。
- [C++](runtime/cpp/README_cn.md)：原生依赖、生命周期、构建执行及验证范围。
- [转换](conversion/README_cn.md)：权重边界、校准、实际编译输入及缺口。
- [评估](evaluator/README_cn.md)：数组格式、协议、指标定义与历史表格。

API 集成使用 `RuntimeModelRunner` 加 `Yolo26DepthTask`，`predict` 严格串联
pre_process → forward → post_process，不另写一套处理。
自转换制品显式组合 `--converted-model`、`--model-path` 及精确 `--asset-id` 契约参照。
新文件标为 user-converted，不继承发布方摘要或实测精度；S lite 校准系数必须与声明权重一致。

[归档 X5](../../../platforms/x5/samples/vision/yolo26_depth/README.md) 和
[归档 S](../../../platforms/s/samples/vision/yolo26_depth/README.md)保留原实现与记录，用于溯源，不是统一入口。

<a id="license"></a>
## 许可证

代码遵循仓库[许可证](../../../LICENSE)。Ultralytics 权重和 SUN RGB-D 数据保留各自条款，本目录不附带二者。
