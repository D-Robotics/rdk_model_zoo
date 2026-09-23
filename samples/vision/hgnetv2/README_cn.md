# HGNetV2 图像分类

<a id="overview"></a>
## 概述

HGNetV2 是用于视觉任务的卷积骨干网络；本 sample 提供 b0–b4 的 ImageNet-1k 分类入口。

[PP-HGNetV2](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/models/ImageNet1k/PP-HGNetV2.md)

输入一张 BGR 图像，输出 ImageNet-1k Top-K 类别 ID、分数和可选标签。统一 Python 任务复用已有分类实现，按前处理、推理、后处理组织；标签读取、绘图和文件输出由 CLI 负责。

<a id="support-matrix"></a>
## 支持矩阵

| 目标 | 变体 | Python | C++ |
| --- | --- | --- | --- |
| x5 | b0 | supported-not-run | not-supported |
| x5 | b1 | supported-not-run | not-supported |
| x5 | b2 | supported-not-run | not-supported |
| x5 | b3 | supported-not-run | not-supported |
| x5 | b4 | supported-not-run | not-supported |
| s100 | b0 | not-supported | not-supported |
| s100 | b1 | not-supported | not-supported |
| s100 | b2 | not-supported | not-supported |
| s100 | b3 | not-supported | not-supported |
| s100 | b4 | not-supported | not-supported |
| s100p | b0 | not-supported | not-supported |
| s100p | b1 | not-supported | not-supported |
| s100p | b2 | not-supported | not-supported |
| s100p | b3 | not-supported | not-supported |
| s100p | b4 | not-supported | not-supported |
| s600 | b0 | not-supported | not-supported |
| s600 | b1 | not-supported | not-supported |
| s600 | b2 | not-supported | not-supported |
| s600 | b3 | not-supported | not-supported |
| s600 | b4 | not-supported | not-supported |

`supported-not-run` 表示已有统一实现与发布制品，但本轮板测未执行；S 系列无对应制品，所有目标均无 C++ 实现。[主机验证记录](../../../docs/releases/unified-migration/2026-09-22-b4-classification-review.md)不替代板端证据。

固定源：rdk_x5 @ac115717197920355fc390bb04299b20e6436864。此 sample 没有 C++ 运行交付。CLI 小写 ID 映射到准确发布文件名；文件名大小写保留。

<a id="prerequisites"></a>
## 前提

使用完整仓库检出。X5 推理需要匹配的板端镜像与 `hbm_runtime`；主机依赖按下方装入虚拟环境。SciPy 仅用于保留源实现的对照测试，统一推理不依赖它。

已验证的本地主机环境：Python 3.14.7、NumPy 2.5.3、OpenCV 4.14.0、PyYAML 6.0.3、SciPy 1.18.1。该组合仅用于主机回归，不是板端依赖版本承诺。计划验证 X5 4GB/8GB；板端系统镜像、Python、SDK 的准确版本及最低内存仍待实测登记。磁盘需容纳仓库、所选模型与输出，本轮未测最低容量。原生推理不需要 OE；重新转换的工具链和数据前提见 conversion 文档。

```bash
# cwd: repository root
python3 -m venv .venv-hgnetv2
source .venv-hgnetv2/bin/activate
python3 -m pip install -r samples/vision/hgnetv2/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## 快速开始

在 X5 仓库根执行。下载成功时退出码 0 并打印观测哈希；推理成功时退出码 0 并打印五条结果。推理不会自动下载模型。

```bash
# cwd: repository root
bash samples/vision/hgnetv2/model/download.sh x5 b0
python3 samples/vision/hgnetv2/runtime/python/main.py \
  --target x5 --variant b0 \
  --test-img samples/vision/hgnetv2/test_data/sandbar.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

<a id="expected-results"></a>
## 预期结果

默认变体 `b0` 保留源入口选择，其余变体 `b1`、`b2`、`b3`、`b4` 须显式指定。分数沿用源 softmax 策略，完全平局时按 ID 升序稳定排序。`sandbar.JPEG` 仅作功能输入，不代表数据集精度；统一实现板端结果尚未产生。仅指定 `--img-save-path` 才保存文件。

<a id="performance"></a>
## 性能数据

历史源数据未在本轮重测。完整列与计时条件见[评测说明](evaluator/README_cn.md#reference-results)，单线程延迟与多线程 FPS 不能按倒数直接比较。

<a id="directory"></a>
## 目录

`model/`：制品与下载；`runtime/python/`：原生 CLI、任务与运行器；`conversion/`：5 份保留 PTQ 配置；`evaluator/`：递归目录 CSV 评测、功能检查与历史基准；`test_data/`：`sandbar.JPEG` 输入及随附资源；`tests/`：主机回归。

<a id="entry-points"></a>
## 入口

[Model](model/README_cn.md) · [Python](runtime/python/README_cn.md) · [Conversion](conversion/README_cn.md) · [Evaluation](evaluator/README_cn.md)

旧 `platforms/x5/samples/vision/hgnetv2` 入口仍是原始实现，供基线对照，未改成转发层。新集成使用本目录；不承诺所有旧内部 Python 导入兼容。

<a id="license"></a>
## 许可

Python 代码遵循 Apache-2.0。转换材料保留各文件原声明，上游模型/权重遵循其各自许可；这里不新增权重许可声明。
