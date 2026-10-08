[English](README.md) | 简体中文

# VargConvNet 图像分类

VargConvNet 是轻量卷积图像分类网络。

<a id="overview"></a>

## 概述

VargConvNet 是面向边缘设备的轻量级卷积分类模型，用于 ImageNet-1k 图像分类。RDK X5 sample 提供 packed-NV12 `.bin` 模型和基于 `hbm_runtime` 的 Python 运行时。

`classify.py` 定义图像输入与 1,000 类分类分数输出。替换模型文件前，按[模型转换](conversion/README_cn.md)核对输入输出。

输入一张 BGR 图像，输出 ImageNet-1k Top-K 类别 ID、分数和可选标签。
`VargConvNetClassifier` 类执行由 `predict` 串联的 `preprocess → infer → postprocess`
流程（标签读取、绘图和文件输出由 CLI 层负责，见
[runtime/python/README_cn.md](runtime/python/README_cn.md)）。

<a id="directory"></a>
## 目录结构

```text
vargconvnet/
├── conversion/  # 导出与量化配置
├── evaluator/  # 评估程序与指标
├── model/  # 模型文件与下载脚本
├── runtime/  # 推理程序
├── test_data/  # 示例输入
├── tests/  # 自动化测试
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
└── requirements-host.txt  # Python 依赖
```

<a id="support-matrix"></a>
## 支持矩阵

| 目标 | 变体 | Python | C++ |
| --- | --- | --- | --- |
| x5 | vargconvnet | supported | not-supported |
| s100 | vargconvnet | not-supported | not-supported |
| s100p | vargconvnet | not-supported | not-supported |
| s600 | vargconvnet | not-supported | not-supported |

S 系列无对应制品，所有目标均无 C++ 实现。

CLI 小写变体 ID 映射到准确发布文件名；文件名大小写保持不变。

<a id="prerequisites"></a>
## 前提

使用完整仓库检出。X5 推理需要匹配的板端镜像与 `hbm_runtime`；主机
依赖按下方装入虚拟环境。SciPy 仅用于主机对照测试，板端推理不依赖
它。原生推理不需要 OE 工具链；重新转换的前提见
[conversion](conversion/README_cn.md) 文档。

```bash
# cwd: repository root
python3 -m venv .venv-vargconvnet
source .venv-vargconvnet/bin/activate
python3 -m pip install -r samples/vision/vargconvnet/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## 快速体验

在 X5 上从仓库根目录执行。下载成功时退出码 0 并打印观测摘要；推理成功时
退出码 0 并打印五条结果。推理不会自动下载模型。

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

唯一已发布变体为 `vargconvnet`，默认即选中。推理打印 Top-5 ID、
softmax 分数与标签。完全平局按类别 ID 升序。随附图片仅作功能输入。
仅指定 `--img-save-path` 才写文件。

下图为 X5 发布的参考运行效果：demo 将 Top-5 排名叠画在随附测试图上，
rank 1 为 class 37（box turtle, box tortoise），分数 0.8582。

![X5 参考推理结果：box turtle 测试图与 Top-5 叠加，rank 1 为 class 37（box turtle, box
tortoise），分数 0.8582](./test_data/inference.png)

<a id="performance"></a>
## 性能数据

数据集精度与延迟的测量流程见[评估说明](evaluator/README_cn.md#reference-results)。

<a id="entry-points"></a>
## 入口

[模型](model/README_cn.md) · [Python 运行时](runtime/python/README_cn.md) ·
[转换](conversion/README_cn.md) · [评估](evaluator/README_cn.md)

<a id="license"></a>
## 许可

Python 代码遵循 Apache-2.0。转换材料保留各文件原声明；上游模型/权重
遵循其各自许可；再分发前请核对上游许可条款。
