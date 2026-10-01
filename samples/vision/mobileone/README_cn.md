# MobileOne 图像分类

<a id="overview"></a>

## 概述

MobileOne 是面向低时延边缘部署的轻量级 CNN 骨干网络。模型通过结构重参数化，在训练阶段保留多分支表达能力，在推理阶段收敛为更简洁的部署结构。

- **论文**: [MobileOne: An Improved One millisecond Mobile Backbone](http://arxiv.org/abs/2206.04040)
- **参考实现**: [apple/ml-mobileone](https://github.com/apple/ml-mobileone)

源 README 提炼的特性：

- **结构重参数化**——训练期的多分支模块（k 个并行卷积分支、逐分支
  BN 与恒等分支，激活用 ReLU 或 SE-ReLU）在推理期融合为每个 block 单条
  卷积的部署友好结构。
- **低时延骨干**——面向移动与嵌入式部署，吞吐能力强。
- **变体缩放**——提供 `S0` 到 `S4` 多个发布变体，过参数化因子 `k`
  按变体调优。
- **分类输出**——产出 ImageNet-1k 标签的 Top-K 类别 ID 与置信度。

![MobileOne block：训练期多分支结构经重参数化得到推理期单卷积](./test_data/MobileOne_architecture.png)

*图（上游论文 Fig. 3）：MobileOne block 有两种结构——训练期带可重参数
化分支（左），推理期把分支折叠为单个 3×3 / 1×1 卷积（右）。恢复自
X5 源 README（rdk_x5
@ac115717197920355fc390bb04299b20e6436864）；训练/推理双结构也解释了
部署制品为何是已重参数化的 INT8 s0–s4 变体（224×224 NV12，见
[支持与实测矩阵](#support-matrix)）。*

输入一张 BGR 图像，输出 ImageNet-1k Top-K 类别 ID、分数和可选标签。统一 Python 任务复用已有分类实现，按前处理、推理、后处理组织；标签读取、绘图和文件输出由 CLI 负责。

<a id="support-matrix"></a>
## 支持矩阵

| 目标 | 变体 | Python | C++ |
| --- | --- | --- | --- |
| x5 | s0 | supported-not-run | not-supported |
| x5 | s1 | supported-not-run | not-supported |
| x5 | s2 | supported-not-run | not-supported |
| x5 | s3 | supported-not-run | not-supported |
| x5 | s4 | supported-not-run | not-supported |
| s100 | s0 | not-supported | not-supported |
| s100 | s1 | not-supported | not-supported |
| s100 | s2 | not-supported | not-supported |
| s100 | s3 | not-supported | not-supported |
| s100 | s4 | not-supported | not-supported |
| s100p | s0 | not-supported | not-supported |
| s100p | s1 | not-supported | not-supported |
| s100p | s2 | not-supported | not-supported |
| s100p | s3 | not-supported | not-supported |
| s100p | s4 | not-supported | not-supported |
| s600 | s0 | not-supported | not-supported |
| s600 | s1 | not-supported | not-supported |
| s600 | s2 | not-supported | not-supported |
| s600 | s3 | not-supported | not-supported |
| s600 | s4 | not-supported | not-supported |

`supported-not-run` 表示已有统一实现与发布制品，但本轮板测未执行；S 系列无对应制品，所有目标均无 C++ 实现。[主机验证记录](../../../docs/releases/unified-migration/2026-09-22-b4-classification-review.md)不替代板端证据。

固定源：rdk_x5 @ac115717197920355fc390bb04299b20e6436864。此 sample 没有 C++ 运行交付。CLI 小写 ID 映射到准确发布文件名；文件名大小写保留。

<a id="prerequisites"></a>
## 前提

使用完整仓库检出。X5 推理需要匹配的板端镜像与 `hbm_runtime`；主机依赖按下方装入虚拟环境。SciPy 仅用于保留源实现的对照测试，统一推理不依赖它。

已验证的本地主机环境：Python 3.14.7、NumPy 2.5.3、OpenCV 4.14.0、PyYAML 6.0.3、SciPy 1.18.1。该组合仅用于主机回归，不是板端依赖版本承诺。计划验证 X5 4GB/8GB；板端系统镜像、Python、SDK 的准确版本及最低内存仍待实测登记。磁盘需容纳仓库、所选模型与输出，本轮未测最低容量。原生推理不需要 OE；重新转换的工具链和数据前提见 conversion 文档。

```bash
# cwd: repository root
python3 -m venv .venv-mobileone
source .venv-mobileone/bin/activate
python3 -m pip install -r samples/vision/mobileone/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## 快速开始

在 X5 仓库根执行。下载成功时退出码 0 并打印观测哈希；推理成功时退出码 0 并打印五条结果。推理不会自动下载模型。

```bash
# cwd: repository root
bash samples/vision/mobileone/model/download.sh x5 s0
python3 samples/vision/mobileone/runtime/python/main.py \
  --target x5 --variant s0 \
  --test-img samples/vision/mobileone/test_data/tiger_beetle.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

<a id="expected-results"></a>
## 预期结果

默认变体 `s0` 保留源入口选择，其余变体 `s1`、`s2`、`s3`、`s4` 须显式指定。分数沿用源 softmax 策略，完全平局时按 ID 升序稳定排序。`tiger_beetle.JPEG` 仅作功能输入，不代表数据集精度；统一实现板端结果尚未产生。仅指定 `--img-save-path` 才保存文件。

供参考：X5 源 README（rdk_x5
@ac115717197920355fc390bb04299b20e6436864，旧版 Python 入口）用下面的
截图演示运行效果：旧版 `result.jpg` 绘制把 Top-5 排名叠在图上，随附
`tiger_beetle.JPEG` 的 rank 1 为 class 300（tiger beetle）。这是源交付
中的历史截图，不是本仓库当前入口的运行结果。

![X5 源 README 的历史推理截图：tiger beetle 测试图与旧版 Top-5 叠加，
rank 1 为 class 300（tiger beetle）](./test_data/inference.png)

<a id="performance"></a>
## 性能数据

历史源数据未在本轮重测。完整列与计时条件见[评测说明](evaluator/README_cn.md#reference-results)，单线程延迟与多线程 FPS 不能按倒数直接比较。

<a id="directory"></a>
## 目录

`model/`：制品与下载；`runtime/python/`：原生 CLI、任务与运行器；`conversion/`：5 份原样 PTQ 配置；`evaluator/`：检查与历史基准；`test_data/`：`tiger_beetle.JPEG` 输入及随附资源；`tests/`：主机回归。

<a id="entry-points"></a>
## 入口

[Model](model/README_cn.md) · [Python](runtime/python/README_cn.md) · [Conversion](conversion/README_cn.md) · [Evaluation](evaluator/README_cn.md)

旧 `platforms/x5/samples/vision/mobileone` 入口仍是原始实现，供基线对照，未改成转发层。新集成使用本目录；不承诺所有旧内部 Python 导入兼容。

<a id="license"></a>
## 许可

源 Python 文件的 Apache-2.0 来源保留。转换 YAML 原有专有声明逐字保留；仓库许可不覆盖这些声明或上游权重许可。分发转换材料、权重前须核对对应声明。
