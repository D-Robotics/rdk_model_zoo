# VargConvNet 图像分类

<a id="overview"></a>
## 概述

VargConvNet 是面向边缘设备的轻量级卷积分类模型，用于 ImageNet-1k 图像分类。RDK X5 sample 提供 packed-NV12 `.bin` 模型和基于 `hbm_runtime` 的 Python 运行时。

[固定源说明](../../../platforms/x5/samples/vision/vargconvnet/README_cn.md)：源材料未提供独立论文或上游仓库链接。

输入一张 BGR 图像，输出 ImageNet-1k Top-K 类别 ID、分数和可选标签。统一 Python 任务复用已有分类实现，按前处理、推理、后处理组织；标签读取、绘图和文件输出由 CLI 负责。

<a id="support-matrix"></a>
## 支持矩阵

| 目标 | 变体 | Python | C++ |
| --- | --- | --- | --- |
| x5 | vargconvnet | supported-not-run | not-supported |
| s100 | vargconvnet | not-supported | not-supported |
| s100p | vargconvnet | not-supported | not-supported |
| s600 | vargconvnet | not-supported | not-supported |

`supported-not-run` 表示已有统一实现与发布制品，但本轮板测未执行；S 系列无对应制品，所有目标均无 C++ 实现。[主机验证记录](../../../docs/releases/unified-migration/2026-09-22-b4-classification-review.md)不替代板端证据。

固定源：rdk_x5 @ac115717197920355fc390bb04299b20e6436864。此 sample 没有 C++ 运行交付。CLI 小写 ID 映射到准确发布文件名；文件名大小写保留。

<a id="prerequisites"></a>
## 前提

使用完整仓库检出。X5 推理需要匹配的板端镜像与 `hbm_runtime`；主机依赖按下方装入虚拟环境。SciPy 仅用于保留源实现的对照测试，统一推理不依赖它。

已验证的本地主机环境：Python 3.14.7、NumPy 2.5.3、OpenCV 4.14.0、PyYAML 6.0.3、SciPy 1.18.1。该组合仅用于主机回归，不是板端依赖版本承诺。计划验证 X5 4GB/8GB；板端系统镜像、Python、SDK 的准确版本及最低内存仍待实测登记。磁盘需容纳仓库、所选模型与输出，本轮未测最低容量。原生推理不需要 OE；重新转换的工具链和数据前提见 conversion 文档。

```bash
# cwd: repository root
python3 -m venv .venv-vargconvnet
source .venv-vargconvnet/bin/activate
python3 -m pip install -r samples/vision/vargconvnet/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## 快速开始

在 X5 仓库根执行。下载成功时退出码 0 并打印观测哈希；推理成功时退出码 0 并打印五条结果。推理不会自动下载模型。

```bash
# cwd: repository root
bash samples/vision/vargconvnet/model/download.sh x5 vargconvnet
python3 samples/vision/vargconvnet/runtime/python/main.py \
  --target x5 --variant vargconvnet \
  --test-img samples/vision/vargconvnet/test_data/box_turtle.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

<a id="expected-results"></a>
## 预期结果

唯一发布变体为 `vargconvnet`，默认即选中它。推理打印 Top-5 ID、softmax 分数与标签。完全平局按 ID 升序排序。随附图片用于功能检查；板测 not-run。仅指定 `--img-save-path` 才写文件。

<a id="performance"></a>
## 性能数据

固定源没有提供发布基准表，本次迁移未测数据集精度或延迟。

<a id="directory"></a>
## 目录

`model/`：制品与下载；`runtime/python/`：原生 CLI、任务与运行器；`conversion/`：0 份保留 PTQ 配置；`evaluator/`：检查与历史基准；`test_data/`：`box_turtle.JPEG` 输入及随附资源；`tests/`：主机回归。

<a id="entry-points"></a>
## 入口

[Model](model/README_cn.md) · [Python](runtime/python/README_cn.md) · [Conversion](conversion/README_cn.md) · [Evaluation](evaluator/README_cn.md)

旧 `platforms/x5/samples/vision/vargconvnet` 入口仍是原始实现，供基线对照，未改成转发层。新集成使用本目录；不承诺所有旧内部 Python 导入兼容。

<a id="license"></a>
## 许可

Python 代码遵循 Apache-2.0。转换材料保留各文件原声明，上游模型/权重遵循其各自许可；这里不新增权重许可声明。
