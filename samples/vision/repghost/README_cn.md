# RepGhost 图像分类

RepGhost 通过结构重参数化减少轻量 CNN 中显式特征拼接的开销。

[English README](README.md)

<a id="overview"></a>

## 概述

RepGhost 是面向硬件高效部署的轻量级 CNN 模型家族，通过将特征空间中的显式特征复用转移到权重空间中的重参数化复用，减少 `Concat` 带来的硬件开销，同时保持较好的分类性能。

- **论文**: [RepGhost: A Hardware-Efficient Ghost Module via Re-parameterization](https://arxiv.org/abs/2211.06088)
- **参考实现**: [ChengpengChen/RepGhost](https://github.com/ChengpengChen/RepGhost)

核心特性：

- **结构重参数化**——把训练期的复杂分支转换为推理期的高效结构。
- **隐式特征复用**——把 GhostNet 式的特征复用从特征空间（`Concat`）
  移到权重空间，避免昂贵的内存拷贝。
- **硬件效率**——减少内存拷贝开销，提升边缘设备部署效率。
- **变体缩放**——提供 `100` 到 `200` 多个发布变体。

![RepGhost bottleneck 与 Ghost bottleneck 对比：训练期 add 分支在推理
期融合](./test_data/RepGhost_architecture.png)

*图（上游论文 Fig. 4）：(a) 带 `Concat` 显式特征复用的 Ghost
bottleneck；(b) 训练期 RG-bneck——复用经 `add` 分支移入权重空间；
(c) 推理期 RG-bneck——分支已融合消失。图中为上游结构，实际部署
制品是 INT8 量化的 100–200 变体（224×224 NV12，见
[支持范围](#support-matrix)）。*

输入一张 BGR 图像，输出 ImageNet-1k Top-K 类别 ID、分数和可选标签。
`RepGhostClassifier` 类执行由 `predict` 串联的 `preprocess → infer → postprocess`
流程（标签读取、绘图和文件输出由 CLI 层负责，见
[runtime/python/README_cn.md](runtime/python/README_cn.md)）。

<a id="directory"></a>
## 目录结构

```text
repghost/
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
| x5 | 100 | supported | not-supported |
| x5 | 111 | supported | not-supported |
| x5 | 130 | supported | not-supported |
| x5 | 150 | supported | not-supported |
| x5 | 200 | supported | not-supported |
| s100 | 100 | not-supported | not-supported |
| s100 | 111 | not-supported | not-supported |
| s100 | 130 | not-supported | not-supported |
| s100 | 150 | not-supported | not-supported |
| s100 | 200 | not-supported | not-supported |
| s100p | 100 | not-supported | not-supported |
| s100p | 111 | not-supported | not-supported |
| s100p | 130 | not-supported | not-supported |
| s100p | 150 | not-supported | not-supported |
| s100p | 200 | not-supported | not-supported |
| s600 | 100 | not-supported | not-supported |
| s600 | 111 | not-supported | not-supported |
| s600 | 130 | not-supported | not-supported |
| s600 | 150 | not-supported | not-supported |
| s600 | 200 | not-supported | not-supported |

S 系列无对应制品，所有目标均无 C++ 实现。

按支持矩阵选择 Python 运行时与目标。

<a id="prerequisites"></a>
## 前提

使用完整仓库检出。X5 推理需要匹配的板端镜像与 `hbm_runtime`；主机
依赖按下方装入虚拟环境。SciPy 仅用于主机对照测试，板端推理不依赖
它。原生推理不需要 OE 工具链；重新转换的前提见
[conversion](conversion/README_cn.md) 文档。

```bash
# cwd: repository root
python3 -m venv .venv-repghost
source .venv-repghost/bin/activate
python3 -m pip install -r samples/vision/repghost/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## 快速体验

在 X5 上从仓库根目录执行。下载成功时退出码 0 并打印观测摘要；推理成功时
退出码 0 并打印五条结果。推理不会自动下载模型。

```bash
# cwd: repository root
bash samples/vision/repghost/model/download.sh x5 100
python3 samples/vision/repghost/runtime/python/main.py \
  --target x5 --variant 100 \
  --test-img samples/vision/repghost/test_data/ibex.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

<a id="expected-results"></a>
## 预期结果

默认变体为 `100`；`111`、`130`、`150`、`200` 须显式指定。分数采用
softmax，完全平局时按类别 ID 升序稳定排序。使用随附图片完成单图功能
检查；数据集精度按评测指南将每张图的真值类别 ID 与 Top-1 结果比较。
仅指定 `--img-save-path` 才保存文件。

下图为 X5 发布的参考运行效果：demo 将 Top-5 排名叠画在随附测试图上，
rank 1 为 class 350（ibex, Capra ibex）。

![X5 参考推理结果：ibex 测试图与 Top-5 叠加，rank 1 为 class 350（ibex, Capra ibex）](./test_data/inference.png)

<a id="performance"></a>
## 性能数据

已发布性能记录：完整列与计时条件见
[评估说明](evaluator/README_cn.md#reference-results)。单线程延迟与多线程 FPS 采用不同的并发方式，
二者不能直接互相取倒数。比较延迟与 FPS 时，应使用相同线程数、并发提交方式和 BPU 利用率。

<a id="entry-points"></a>
## 入口

[模型](model/README_cn.md) · [Python 运行时](runtime/python/README_cn.md) ·
[转换](conversion/README_cn.md) · [评估](evaluator/README_cn.md)

<a id="license"></a>
## 许可

Python 文件采用 Apache-2.0。转换 YAML 的许可见文件声明；
仓库许可不覆盖这些声明或上游权重许可。再分发转换材料或权重前请核对
适用声明。
